# -*- coding: utf-8 -*-
"""verify_routing_v11.py — SolAttn routing v1.1 修改验证（2026-08-21）。

验证点（对照逐行 Python 参考实现，CPU 上跑小规模）：
  1. 无 nnz_s（全尺寸）时：新 csr_from_sel == 旧逻辑（逐行参考）
  2. nnz_s=NB//2 时：每行 cnt <= nnz_s，off = 前 nnz_s 个选中块的 col
  3. 非连续输入 q3/k3：routing 结果与连续输入一致
  4. GPU 上小规模端到端（build_sparse_csr + 无 .any()/无 .item() 路径）无异常
  5. 密度统计（cap 前后差异）

用法：python verify_routing_v11.py [--gpu]（--gpu 在 V100 上跑 CUDA 路径）
"""
import argparse
import math
import sys

import torch

sys.path.insert(0, r"E:\xxx\ComfyUI-MiniMaxH3-SolAttn-V100")
import routing


def ref_csr(sel, block=64, nnz_s=None):
    """逐行 Python 参考实现（旧逻辑）：sel [NQB,H,NB] bool -> (cnt, off)"""
    NQB, H, NB = sel.shape
    b_h = NQB * H
    sel_perm = sel.permute(1, 0, 2).reshape(b_h, NB)
    cnt = sel_perm.sum(dim=1)
    slot = NB if nnz_s is None else min(nnz_s, NB)
    off = torch.zeros(b_h, slot, dtype=torch.int32)
    for row in range(b_h):
        cols = [c for c in range(NB) if sel_perm[row, c]][:slot]
        for j, c in enumerate(cols):
            off[row, j] = c * block
    return cnt.to(torch.int32), off


def check(cond, msg):
    if not cond:
        print(f"[FAIL] {msg}")
        sys.exit(1)
    print(f"[ OK ] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="GPU 端到端验证")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"device = {dev}")

    # ---- 1/2: CPU 小规模正确性（随机稀疏掩码 + 真路由）----
    for (NQB, H, NB) in [(20, 8, 34), (37, 4, 60), (10, 2, 130)]:
        sel = torch.rand(NQB, H, NB) < 0.26
        # 1) 全尺寸等价
        cnt1, off1, _, _ = routing.csr_from_sel(sel)
        cnt_r, off_r = ref_csr(sel)
        check(torch.equal(cnt1, cnt_r) and torch.equal(off1, off_r),
              f"shape={sel.shape} 全尺寸 == 参考")
        # 2) cap 生效
        slot = max(1, NB // 2)
        cnt2, off2, _, _ = routing.csr_from_sel(sel, nnz_s=slot)
        assert off2.shape[1] == slot, f"off 宽度 {off2.shape[1]} != {slot}"
        check(bool((cnt2 <= slot).all()), f"shape={sel.shape} cap 后每行 cnt<=slot")
        cnt_r2, off_r2 = ref_csr(sel, nnz_s=slot)
        check(torch.equal(cnt2, cnt_r2) and torch.equal(off2, off_r2),
              f"shape={sel.shape} cap={slot} == 参考")

    # ---- 3: 非连续输入路由一致性（同一份数据的连续/非连续视图）----
    S, H, D = 3000, 8, 128
    q_base = torch.randn(D, H, S, dtype=torch.float16).permute(2, 1, 0)  # [S,H,D] 非连续
    k_base = torch.randn(D, H, S, dtype=torch.float16).permute(2, 1, 0)
    q_nc, k_nc = q_base, k_base
    q_c, k_c = q_base.contiguous(), k_base.contiguous()
    a1 = routing.build_sparse_csr(q_c, k_c, tau=1.0, scale=D ** -0.5)
    a3 = routing.build_sparse_csr(q_nc, k_nc, tau=1.0, scale=D ** -0.5)
    check(q_nc.is_contiguous() == False and k_nc.is_contiguous() == False, "输入确实非连续")
    check(all(torch.equal(x, y) for x, y in zip(a1, a3)),
          "非连续输入路由 == 连续输入路由（值一致）")
    dens = a1[0].float().mean() / ((S + 63) // 64)
    print(f"       tau=1.0 密度 ≈ {dens * 100:.1f}%（S={S}, NB={(S + 63) // 64}）")

    # ---- 4: GPU 端到端 ----
    if args.gpu and torch.cuda.is_available():
        qg = q_nc.to("cuda")
        kg = k_nc.to("cuda")
        nnz_s = max(256, (S + 63) // 64 // 2)
        cnt, off, ccnt, cidx = routing.build_sparse_csr(qg, kg, tau=1.0,
                                                         scale=D ** -0.5, nnz_s=nnz_s)
        check(off.shape[1] == min(nnz_s, (S + 63) // 64), f"GPU off 宽度 = {off.shape[1]}")
        check(bool((cnt <= off.shape[1]).all().item()), "GPU 每行 cnt <= slot")
        # 非零 off 在边界内
        check(bool((off >= 0).all().item()) and bool((off < (S + 63) // 64 * 64).all().item()),
              "GPU off 值域合法")
        print(f"       GPU 路由 OK: b_h={cnt.numel()}, slot={off.shape[1]}")
    else:
        print("[skip] GPU 端到端（未指定 --gpu 或无 CUDA）")

    print("\nALL PASS")


if __name__ == "__main__":
    main()
