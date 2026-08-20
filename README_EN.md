# ComfyUI-MiniMaxH3-SolAttn-V100

**Sol-Attn (arXiv 2607.24027) sparse acceleration · single-node attention accelerator for MiniMax H3 on V100 (ComfyUI custom node) · v1.0.0**

> 中文版: [README.md](README.md) · Development/design docs: [DEVELOPMENT.md](DEVELOPMENT.md) / [DEVELOPMENT_EN.md](DEVELOPMENT_EN.md)
>
> **AI-assisted development notice**: This project was developed by the author with the help of an AI assistant (DeepSeek-V4-Flash). The author has no computer-science background; code and docs were co-written with AI. The project is shared as-is under the MIT license; if you run into issues, please open an Issue and we will help within our capabilities.

One node = **embedded FP16Safe (fp16 safety) + Sol-Attn official sparse (keep-or-drop)**. On V100 (sm_70), MiniMax H3 video generation drops from **71-74 s/step to 43 s/step at 480p/10s (~1.7×), with no visible quality loss**.

---

## Credits (what this project references and uses)

Everything here is a **proven, existing implementation** combined together. The acceleration comes entirely from:

| Component | Source | Role |
|---|---|---|
| **Sol-Attn sparse algorithm** | [arXiv 2607.24027](https://arxiv.org/abs/2607.24027) (NVlabs/Sana sol-engine, `techniques/sparse_backends/sol_attn/preprocess.py`) | kc/vc block stats + diag threshold + column-mean routing + keep-or-drop (`routing.py` re-implements the official algorithm; per-head independent routing) |
| **keep-or-drop kernel** | direct source [rwashy/H3-V100](https://github.com/rwashy/H3-V100) (repo is GPL-3.0-only; its FlashAttention CUDA component is BSD 3-Clause — only that BSD component is copied); upstream [flash-attention](https://github.com/Dao-AILab/flash-attention) (Tri Dao) + [Icbears/flash-attention-v100](https://github.com/Icbears/flash-attention-v100) | selected blocks computed exactly, others skipped (`comfy_v100_solattn_cuda`: source in `native/`, or prebuilt pyd from Release) |
| **fp16 NaN safety** | [ComfyUI-MiniMaxH3-FP16Safe](https://github.com/aaalll12322/ComfyUI-MiniMaxH3-FP16Safe) v6.8.0 (logic embedded as `fp16safe.py`) | prescale /16 + deferred fuse + fp32 re-run fallback (**self-contained, no separate FP16Safe install needed**) |
| **Parameter style** | [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) | tau / start_percent / end_percent / dense_blocks / sink parameter family |

---

## Problem

Two core bottlenecks of MiniMax H3 on V100:

1. **Attention dominates** (~79% of per-step time at 480p), while PyTorch SDPA on V100 only reaches ~37T;
2. **fp16 compute NaN** (H3 activations genuinely reach ~5e5, far beyond the fp16 limit ±65504); official dtype support is bf16/fp32 only, and V100 has no bf16 hardware so it falls back to fp32 (4× slower).

Sol-Attn sparse idea: most attention scores are noise — **compute only the few high-value KV blocks exactly** (keep-or-drop) and skip the rest, removing the O(n²) bulk.

---

## Measured performance (user's real ComfyUI, 480p/10s)

| Config | s/step | Quality |
|---|---|---|
| Pure FP16Safe (baseline) | 71-74s | normal |
| Pure SDPA + FP16Safe (dense fallback) | ~70s | normal |
| **Sol-Attn (recommended, `tau=1.0`)** | **43s** | **no visible loss** |

- vs pure FP16Safe: **~1.7×**
- Single attention (S=29650): sparse kernel **4.55×** vs SDPA (route+kernel 2.77×)
- Routing vectorized: **10ms @ S=29650**; kernel consumes non-contiguous inputs (saves 3 copies)

---

## Install

```bash
cd ComfyUI/custom_nodes
# Option 1: git clone (when published to GitHub)
git clone https://github.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100.git
# Option 2: copy the whole folder into custom_nodes/ (kernel needed, see below)
```

**Getting the kernel** (`comfy_v100_solattn_cuda*.pyd`, pick one; the plugin auto-matches any pyd whose name starts with `comfy_v100_solattn_cuda` — no specific Python-version suffix required):
- **Release prebuilt**: download from GitHub Release (Windows + Python 3.12, zero compilation)
- **Build from source**: full source ships in `native/` (incl. CUTLASS), one command:
  ```bash
  cd native && python setup.py build_ext --inplace
  # copy the artifact (comfy_v100_solattn_cuda.cpXXX-win_amd64.pyd, XXX follows the
  # Python version used to build) to the plugin root — the plugin detects it automatically
  ```

Restart ComfyUI. Find **"Sol-Attn (V100)"** under the `sol_attn` category.

**Dependencies**:
- **ComfyUI** (with `comfy/ldm/minimax`, PR #15224)
- **NVIDIA V100 (sm_70)** only (raises at startup on other architectures)
- **No extra Python packages, no extra plugins**: FP16Safe logic embedded (`fp16safe.py`); the CUTLASS kernel source ships in `native/`, prebuilt pyd from Release

---

## Usage

```
H3 model ──> Sol-Attn (V100) ──> sampler (KSampler etc.)
```

The single node performs fp16 safety + sparse. **No separate FP16Safe node needed.**

### Parameters

| Param | Default | Description |
|---|---|---|
| `fp16_safe` | true | Embed FP16Safe (prescale /16 + deferred fuse + fp32 re-run). Disable only if another node provides fp16 safety |
| `tau` | 1.0 | Sparse routing threshold β: higher = sparser/faster. 1.0 ≈ 26% density (V100 fp16, user-verified no visible loss); 1.5 lower density/faster |
| `start_percent` | 0.2 | Run dense before this sampling progress (paper uses 0.2; with turbo 4-step the first step is naturally dense) |
| `end_percent` | 1.0 | Run dense after this sampling progress (1.0 = no trailing dense, matches the measured 43 s/step; 0.9 keeps trailing quality) |
| `min_tokens` | 1024 | Sequences shorter than this stay dense (SDPA) |
| `dense_blocks` | "0-1" | Transformer blocks kept dense, e.g. `"0-1"` = first two, `"0-2,-1"` = first three + last (-1 counts from the end); empty = sparsify all |
| `h3_prefix_tokens` | 0 | H3 text/cond/ref/audio prefix tokens (KV sink; e.g. 634+cond+ref+audio; can stay 0 in turbo scenes) |
| `debug_nan` / `profile` | false | Pass-through FP16Safe NaN detection / timing stats |

### Recommended configs

- **Recommended (fastest at 480p)**: `tau=1.0, start_percent=0.2, end_percent=1.0, dense_blocks="0-1"` → 43 s/step (no visible loss)
- **More aggressive**: `tau=1.2~1.5` (lower density, faster; verify quality yourself)
- **Conservative**: `dense_blocks="0-2,-1"` or `end_percent=0.9` (more layers / trailing dense, sturdier quality, slightly slower)
- **Small resolutions** (608 and below): attention share is low, sparse gains are small — prefer `end_percent=0` (fully dense) or skip the plugin

---

## Principles (summary)

1. **Routing (routing.py, official Sol-Attn algorithm)**: kc block-mean / vc block-sum → diag threshold (analytic projection in key space) → column-mean routing (|neighbor ±1) → CSR mask. Per-head independent; at 26% density rel-L2 is 0.22 vs dense, yet **real videos show no visible loss**.
2. **Kernel**: keep-or-drop sparse kernel (selected blocks computed exactly, others skipped), fp16 + head_dim 128, sm70 CUTLASS.
3. **FP16Safe**: x/16 prescale → qkv → attention → out_proj, unscaled ×16 in fp32; deferred isfinite fuse re-runs the whole forward in fp32 if triggered.

---

## Known limitations

- Verified only on **V100 (sm_70)** + Windows + Python 3.12 (cp312 pyd); other platforms require rebuilding native.
- Sparse routing runs in Python (~10ms @ S=29650) — still optimizable; the kernel itself is fast (4.55×).
- Sparse quality is judged by real video (PSNR/eyes); for sensitive scenes (fine text), raise `dense_blocks` / `end_percent` or lower `tau`.
- Dense fallback = plain SDPA.

---

## Changelog

- **v1.0.0 (2026-08-20) initial release**: single node = embedded FP16Safe (`fp16safe.py`, v6.8.0 logic, self-contained) + Sol-Attn sparse (keep-or-drop, sparse-only kernel). Measured 480p/10s at **43 s/step, no visible loss** (~1.7× vs pure FP16Safe 71-74 s/step). Parameters aligned to kijai style (tau / start_percent / end_percent / min_tokens / dense_blocks / h3_prefix_tokens); dense fallback = plain SDPA; kernel source in `native/` (self-buildable), prebuilt pyd from Release. Tag `[SolAttn-V100][V1.0]`.

---

## Citation

The sparse algorithm used by this project comes from the Sol-Attn paper. If this project helps you, please also cite:

```bibtex
@article{solattn,
  title={Sol-Attn: Training-free Sparse Attention for Accelerating Image and Video Generation},
  author={NVlabs / Sana sol-engine team},
  journal={arXiv preprint arXiv:2607.24027},
  year={2026}
}
```

- Paper: https://arxiv.org/abs/2607.24027
- Official code: <https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends/sol_attn> (Sol-Attn lives on the `sol-engine` branch)
- Project page: https://nvlabs.github.io/Sana/Sol-Attn/
