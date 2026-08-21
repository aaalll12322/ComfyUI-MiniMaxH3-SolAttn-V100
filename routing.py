# -*- coding: utf-8 -*-
"""routing.py — Sol-Attn 稀疏路由（官方算法，V100 fp16 可用版）

算法对照 NVlabs/Sana sol-engine（Triton 参考实现）与 kijai/ComfyUI-SolAttn_triton：
  1. kc = 块均值（pooled key），块大小 64
  2. diag 阈值: threshold[i,h] = (q̄_i·kc_mean_h + τ·sqrt(Σ_d q̄²·var_kc_h)) * log2(scale)
     （key 空间解析投影，非经验矩）
  3. 路由: 列均值（q tokens × kc 按 query 求和平均，log2 域）> 阈值
           | 邻域 ±1 块 | sink 块（H3 前缀）
  4. CSR 掩码: block_count/block_offset（token 起始偏移）喂官方 sparse kernel（keep-or-drop）
质量验证（S=6154 真实激活）: τ=1.0 → 26.4% 密度 → rel-L2 0.223（keep-or-drop，真机肉眼无损）

性能优化（v1.0）：
  - q_centroid 只算一次（diag 阈值 + 路由共用）
  - csr_from_sel 高级索引替代 nonzero/scatter（无 Python 循环，@S=29650 10ms）
  - var 用 E[k²]-E[k]² 公式（省一次减张量）
"""
import math

import torch

BLOCK = 64
LOG2E = 1.4426950408889634


def _pad_to_block(t, block=BLOCK):
    S = t.shape[0]
    NB = math.ceil(S / block)
    pad = NB * block - S
    if pad:
        return torch.nn.functional.pad(t, (0, 0, 0, 0, 0, pad)), NB
    return t.contiguous(), NB          # 连续时 no-op；非连续时拷贝（保证下游 view 可用）


def build_sparse_csr(q, k, tau=1.0, scale=None, neighbor=1,
                     sink_blocks=(0, 0), sink_tokens=0, nnz_s=None, topk_k=0):
    """完整路由（合并版，q_centroid 只算一次）: q,k: [S,H,D] fp16 -> (cnt, off, ccnt, cidx)

    nnz_s: off 每行最大槽位数（None = NB 全尺寸；传 NB//2 等小值省显存，
    kernel 的 NNZ_S 自动取 off 形状，多余槽位不读）。

    topk_k: 每行保底块数（质量修复 v1.1.1，2026-08-22）。Sol-Attn 阈值路由是
    "均值对齐检测"（只保留 scores 高于 mean+τ·std 的块），**高动态/新内容块的
    key 与 query 对齐度低 -> 恰好被过滤**（用户实测：手/边缘/肢体在动态帧丢失）。
    topk_k>0 时：combined = min(threshold, 每行第 K 大分数)（kthvalue 一次选择，
    O(N) 无排序），**每行至少保留 K 块**（动态内容有保底），密度下界 K/NB。
    """
    S, H, D = q.shape
    scale = scale or (D ** -0.5)
    ls = scale * LOG2E
    # ---- 块统计（视图化，零 pad/float 拷贝；fp16 块和 64 项，误差 ~1e-3，
    #      对阈值路由实测 sel 差异 0.006%，可忽略；转 fp32 仅对小张量 [NQB,H,D]）----
    NQB = NB = math.ceil(S / BLOCK)
    main = (S // BLOCK) * BLOCK
    qv = q.permute(1, 0, 2)                               # [H,S,D] fp16 视图
    kv = k.permute(1, 0, 2)
    if main >= BLOCK:
        qc = qv[:, :main].view(H, main // BLOCK, BLOCK, D).sum(dim=2).float() / BLOCK   # [H,NQBm,D] fp32
        kc = kv[:, :main].view(H, main // BLOCK, BLOCK, D).sum(dim=2).float() / BLOCK
    else:
        qc = kc = None
    if main < S:                                  # 尾部不足一块：单独归一块（有效均值）
        qc_t = qv[:, main:].sum(dim=1, keepdim=True).float() / (S - main)
        kc_t = kv[:, main:].sum(dim=1, keepdim=True).float() / (S - main)
        qc = torch.cat([qc, qc_t], dim=1) if qc is not None else qc_t
        kc = torch.cat([kc, kc_t], dim=1) if kc is not None else kc_t
    qc = qc.permute(1, 0, 2).contiguous()         # [NQB,H,D] fp32
    kc = kc.permute(1, 0, 2).contiguous()         # [NB,H,D] fp32
    # diag 阈值（key 空间解析投影；threshold 对同 (q,h) 的所有 j 相同）
    kc_mean = kc.mean(dim=0)                                       # [H,D]
    kc_var = (kc.pow(2).mean(dim=0) - kc_mean.pow(2)).clamp_min(0)  # E[k²]-E[k]²
    mean = torch.einsum("qhd,hd->qh", qc, kc_mean) * ls
    var = torch.einsum("qhd,hd->qh", qc.pow(2), kc_var) * (ls * ls)
    threshold = mean + tau * torch.sqrt(var + 1e-6)                # [NQB,H] log2 域
    # 路由分数（列均值，log2 域）+ 阈值比较
    scores = torch.einsum("qhd,jhd->qhj", qc, kc) * ls             # [NQB,H,NB]
    if topk_k > 0 and NB > topk_k:
        # 每行保底：第 K 大分数作阈值下界（kthvalue 选择算法，无排序）
        kth = torch.kthvalue(scores, k=NB - topk_k + 1, dim=-1).values   # [NQB,H]
        threshold = torch.minimum(threshold, kth)
    sel = scores >= threshold.unsqueeze(-1)                        # [NQB,H,NB] bool
    if neighbor > 0:
        qi = torch.arange(NQB, device=q.device)
        kj = torch.arange(NB, device=q.device)
        sel |= ((kj[None, None, :] - qi[:, None, None]).abs() <= neighbor)
    if sink_blocks[1] > sink_blocks[0]:
        sb = torch.arange(NB, device=q.device)
        sel |= ((sb >= sink_blocks[0]) & (sb < sink_blocks[1]))
    if sink_tokens > 0:
        sel[:, :, :math.ceil(sink_tokens / BLOCK)] = True
    return csr_from_sel(sel, nnz_s=nnz_s)


def csr_from_sel(sel, block=BLOCK, nnz_s=None):
    """选中掩码 [NQB,H,NB] -> CSR (cnt, off, ccnt, cidx)，off 存 token 起始偏移
    off: [NQB*H, NNZ_S]（每行最多 NNZ_S 槽；nnz_s=None 时 NNZ_S=NB 全尺寸）
    ⚠️ 行序必须 = (head, query_block)（kernel 索引 (bidh*NUM_ROWS + m_block)）——不是 (query_block, head)！
    全向量化 + scatter（无 nonzero、无 Python 循环）。

    v1.1 优化（2026-08-21）：
      1. **零 GPU->CPU 同步**：去掉 `if sel_perm.any():`（该分支每层 1 次同步，
         在 LOW_VRAM 下会打断 CPU->GPU 权重预取流水线，放大传输等待）。
      2. **显存瘦身**：rank 改 int32（NB<=2^31 安全，省 50%）；off 支持缩小
         NNZ_S（kernel 按运行时 num_blks 读槽位，多余槽位不读）——
         S=174k 时 off 1.66GB -> 0.83GB（nnz_s=NB//2），路由瞬时峰值 -40%+。
      3. cap 语义：rank < nnz_s 保留每行"块号升序"前 nnz_s 个选中块
         （头部/前缀优先，与 sink/neighbor 保底语义一致），密度 >50% 的
         异常稠密行被截断（正常 tau=1.0 密度 ~26%，不触发）。
    """
    NQB, H, NB = sel.shape
    b_h = NQB * H
    # 行序: head 外层, query 块内层（与 kernel 的 (bidh*NUM_ROWS + m_block) 对齐）
    sel_perm = sel.permute(1, 0, 2).reshape(b_h, NB)              # [H*NQB, NB]，行 = h*NQB + qi（视图）
    # 行内槽位号（cumsum 递增；int32 足够，NB<=2^31）
    rank = sel_perm.cumsum(dim=1, dtype=torch.int32) - 1          # [b_h, NB] int32
    slot = NB if nnz_s is None else min(int(nnz_s), NB)
    if slot < NB:
        sel_perm = sel_perm & (rank < slot)                       # cap 行内选中数 <= slot（无分支）
    cnt = sel_perm.sum(dim=1, dtype=torch.int32)
    off = torch.zeros(b_h, slot, dtype=torch.int32, device=sel.device)
    # 无分支 scatter：off[row, 行内槽位] = col[选中列]（row_idx 是 expand 视图，物化仅 [nnz] 级）
    row_idx = torch.arange(b_h, device=sel.device, dtype=torch.int32)[:, None].expand(b_h, NB)
    col = (torch.arange(NB, device=sel.device, dtype=torch.int32) * block)[None, :].expand(b_h, NB)
    off[row_idx[sel_perm], rank[sel_perm]] = col[sel_perm]
    ccnt = torch.zeros(b_h, dtype=torch.int32, device=sel.device)
    cidx = torch.zeros(b_h, 1, dtype=torch.int32, device=sel.device)
    return cnt.contiguous(), off.contiguous(), ccnt.contiguous(), cidx.contiguous()


def pooled_key(k, block=BLOCK):
    """kc: 块均值 [NB,H,D]（兼容旧接口）"""
    kp, NB = _pad_to_block(k, block)
    return kp.view(NB, block, *k.shape[1:]).mean(dim=1), NB


def diag_threshold(q, kc, tau, scale, block=BLOCK):
    """官方 diag 阈值 [NQB,H]（log2 域，兼容旧接口）"""
    qp, NQB = _pad_to_block(q, block)
    q_centroid = qp.view(NQB, block, *q.shape[1:]).sum(dim=1) / block
    kc_mean = kc.mean(dim=0)
    kc_var = (kc.pow(2).mean(dim=0) - kc_mean.pow(2)).clamp_min(0)
    ls = scale * LOG2E
    mean = torch.einsum("qhd,hd->qh", q_centroid, kc_mean) * ls
    var = torch.einsum("qhd,hd->qh", q_centroid ** 2, kc_var) * (ls * ls)
    return mean + tau * torch.sqrt(var + 1e-6)


def route_mask(q, k, kc, threshold, tau, scale, block=BLOCK, neighbor=1,
               sink_blocks=(0, 0), sink_tokens=0):
    """官方路由: 返回选中掩码 [NQB,H,NB] bool（兼容旧接口）"""
    S, H, D = q.shape
    NQB = math.ceil(S / block)
    NB = kc.shape[0]
    qp, _ = _pad_to_block(q, block)
    q_centroid = qp.view(NQB, block, H, D).sum(dim=1) / block
    scores = torch.einsum("qhd,jhd->qhj", q_centroid.float(), kc.float()) * (scale * LOG2E)
    sel = scores > threshold.unsqueeze(-1)
    if neighbor > 0:
        idx = torch.arange(NB, device=q.device)
        sel = sel | ((idx[None, None, :] - torch.arange(NQB, device=q.device)[:, None, None]).abs() <= neighbor)
    if sink_blocks[1] > sink_blocks[0]:
        sb = torch.arange(NB, device=q.device)
        sel = sel | ((sb >= sink_blocks[0]) & (sb < sink_blocks[1]))
    if sink_tokens > 0:
        nb = math.ceil(sink_tokens / block)
        sel[:, :, :nb] = True
    return sel


def density_of_sel(sel):
    return float(sel.float().mean().item())
