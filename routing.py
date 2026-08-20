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
    return (torch.nn.functional.pad(t, (0, 0, 0, 0, 0, pad)) if pad else t), NB


def build_sparse_csr(q, k, tau=1.0, scale=None, neighbor=1,
                     sink_blocks=(0, 0), sink_tokens=0):
    """完整路由（合并版，q_centroid 只算一次）: q,k: [S,H,D] fp16 -> (cnt, off, ccnt, cidx)"""
    S, H, D = q.shape
    scale = scale or (D ** -0.5)
    ls = scale * LOG2E
    qp, NQB = _pad_to_block(q)
    kp, NB = _pad_to_block(k)
    # 块统计（fp32 计算保证路由精度）
    qc = qp.view(NQB, BLOCK, H, D).sum(dim=1).float() / BLOCK      # [NQB,H,D] 质心
    kc = kp.view(NB, BLOCK, H, D).mean(dim=1).float()              # [NB,H,D] 块均值
    # diag 阈值（key 空间解析投影）
    kc_mean = kc.mean(dim=0)                                       # [H,D]
    kc_var = (kc.pow(2).mean(dim=0) - kc_mean.pow(2)).clamp_min(0)  # E[k²]-E[k]²
    mean = torch.einsum("qhd,hd->qh", qc, kc_mean) * ls
    var = torch.einsum("qhd,hd->qh", qc.pow(2), kc_var) * (ls * ls)
    threshold = mean + tau * torch.sqrt(var + 1e-6)                # [NQB,H] log2 域
    # 路由分数（列均值，log2 域）+ 阈值比较
    scores = torch.einsum("qhd,jhd->qhj", qc, kc) * ls             # [NQB,H,NB]
    sel = scores > threshold.unsqueeze(-1)                         # [NQB,H,NB] bool
    if neighbor > 0:
        qi = torch.arange(NQB, device=q.device)
        kj = torch.arange(NB, device=q.device)
        sel |= ((kj[None, None, :] - qi[:, None, None]).abs() <= neighbor)
    if sink_blocks[1] > sink_blocks[0]:
        sb = torch.arange(NB, device=q.device)
        sel |= ((sb >= sink_blocks[0]) & (sb < sink_blocks[1]))
    if sink_tokens > 0:
        sel[:, :, :math.ceil(sink_tokens / BLOCK)] = True
    return csr_from_sel(sel)


def csr_from_sel(sel, block=BLOCK):
    """选中掩码 [NQB,H,NB] -> CSR (cnt, off, ccnt, cidx)，off 存 token 起始偏移
    off: [NQB*H, NB]（每行最多 NB 槽，NNZ_S=NB）
    ⚠️ 行序必须 = (head, query_block)（kernel 索引 (bidh*NUM_ROWS + m_block)）——不是 (query_block, head)！
    全向量化 + scatter_（无 nonzero、无 Python 循环）。
    """
    NQB, H, NB = sel.shape
    b_h = NQB * H
    # 行序: head 外层, query 块内层（与 kernel 的 (bidh*NUM_ROWS + m_block) 对齐）
    sel_perm = sel.permute(1, 0, 2).reshape(b_h, NB)              # [H*NQB, NB]，行 = h*NQB + qi
    cnt = sel_perm.sum(dim=1).to(torch.int32)
    # 行内槽位号（cumsum 递增；只对选中列生效）
    rank = sel_perm.cumsum(dim=1) - 1                             # [b_h, NB] int64
    col = torch.arange(NB, device=sel.device, dtype=torch.int32) * block
    off = torch.zeros(b_h, NB, dtype=torch.int32, device=sel.device)
    # 只写选中位置：off[row, 槽位] = col[选中列]（高级索引，无污染）
    if sel_perm.any():
        row_idx = torch.arange(b_h, device=sel.device)[:, None].expand(b_h, NB)[sel_perm]
        off[row_idx, rank[sel_perm]] = col.expand(b_h, NB)[sel_perm]
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
