# -*- coding: utf-8 -*-
"""verify_topk_quality.py — SolAttn top-k 保底质量验证（2026-08-22）

用真实激活快照 _h3_sparse_snapshots.pt (q,k,v, S=6154) 对比：
  dense SDPA vs 原 sparse (tau=1.0) vs tau=1.0+topk32 vs tau=0.75+topk32 vs tau=0.75+topk64
指标：rel 误差(vs dense)、密度、路由+kernel 耗时。判断 top-k 保底是否降低误差（质量↑）
且密度开销可控。

用法：python verify_topk_quality.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, r"E:\xxx\ComfyUI-MiniMaxH3-SolAttn-V100")
import routing

torch.ops.load_library(os.path.join(
    r"E:\xxx\ComfyUI-MiniMaxH3-SolAttn-V100", "comfy_v100_solattn_cuda.cp312-win_amd64.pyd"))

H, D = 56, 128


def main():
    torch.cuda.init()
    snap = torch.load(r"E:\xxx\_h3_sparse_snapshots.pt", map_location="cpu", weights_only=True)
    q, k, v = [t.to("cuda") for t in snap[6154]]
    S = q.shape[-2]
    q3 = q.squeeze(0).permute(1, 0, 2)   # [S,H,D] 非连续（与节点一致）
    k3 = k.squeeze(0).permute(1, 0, 2)
    v3 = v.squeeze(0).permute(1, 0, 2)
    scale = D ** -0.5
    print(f"真实激活快照 S={S} H={H} D={D}")

    # dense 参照
    q4 = q3.unsqueeze(0).permute(0, 2, 1, 3)
    k4 = k3.unsqueeze(0).permute(0, 2, 1, 3)
    v4 = v3.unsqueeze(0).permute(0, 2, 1, 3)
    out_dense = torch.nn.functional.scaled_dot_product_attention(
        q4, k4, v4, scale=scale).permute(0, 2, 1, 3).reshape(1, S, H * D).squeeze(0).float()

    def run(tau, topk):
        nnz_s = max(256, (S + 63) // 64 // 2)
        cnt, off, ccnt, cidx = routing.build_sparse_csr(
            q3, k3, tau=tau, scale=scale, sink_tokens=0, nnz_s=nnz_s, topk_k=topk)
        cu = torch.tensor([0, S], dtype=torch.int32, device="cuda")
        t0 = time.perf_counter()
        out, _ = torch.ops.comfy_v100_solattn_cuda.varlen_fwd_sparse(
            q3, k3, v3, None, cu, cu, cnt, off, ccnt, cidx, S, S, scale)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        dens = float(cnt.float().mean().item()) / ((S + 63) // 64)
        out_f = out.reshape(S, H * D).float()
        rel = (out_f - out_dense).abs().mean().item() / out_dense.abs().mean().item()
        return rel, dens, dt

    print(f"\n{'配置':<28s} {'rel(vs dense)':>14s} {'密度':>7s} {'kernel+路由':>10s}")
    print("-" * 66)
    for tau, topk in [(1.0, 0), (1.0, 32), (0.75, 32), (0.75, 64), (0.75, 0)]:
        rel, dens, dt = run(tau, topk)
        tag = f"tau={tau} topk={topk}" + ("  ← 原版" if (tau == 1.0 and topk == 0) else
                                          "  ← 推荐" if (tau == 0.75 and topk == 32) else "")
        print(f"{tag:<28s} {rel:14.4f} {dens*100:6.1f}% {dt:8.1f}ms")

    # 质量分布：topk 对"低分块"（动态候选）的覆盖
    print("\n[质量机制] topk 保底覆盖了多少阈值路由不选中的块：")
    for tau, topk in [(1.0, 32), (0.75, 64)]:
        nnz_s = max(256, (S + 63) // 64 // 2)
        cnt, off, _, _ = routing.build_sparse_csr(
            q3, k3, tau=tau, scale=scale, sink_tokens=0, nnz_s=nnz_s, topk_k=0)
        cnt_t, off_t, _, _ = routing.build_sparse_csr(
            q3, k3, tau=tau, scale=scale, sink_tokens=0, nnz_s=nnz_s, topk_k=topk)
        extra = (cnt_t - cnt).clamp_min(0)
        print(f"  tau={tau} topk={topk}: 保底额外选中块 均值 {extra.float().mean():.1f}/行 "
              f"（原密度 {cnt.float().mean()/((S+63)//64)*100:.1f}% -> {cnt_t.float().mean()/((S+63)//64)*100:.1f}%）")


if __name__ == "__main__":
    main()
