# -*- coding: utf-8 -*-
"""fp16safe.py — MiniMax H3 fp16 安全实现（内嵌自 ComfyUI-MiniMaxH3-FP16Safe v6.8.0，MIT）。

本模块将 FP16Safe 的核心逻辑（prescale /16 + deferred isfinite 熔断 + fp32 重跑兜底、
structure-clone、fp32 residual stream + fp16 Tensor-Core 计算）内嵌到 Sol-Attn 插件，
使本插件自包含：**无需单独安装 ComfyUI-MiniMaxH3-FP16Safe**。

与上游的差异：仅移除 ComfyUI 节点注册（避免与上游插件同名节点冲突），patch 能力不变。
上游：https://github.com/aaalll12322/ComfyUI-MiniMaxH3-FP16Safe（v6.8.0，MIT）
"""
# ComfyUI-MiniMaxH3-FP16Safe v6.8.0
#
# Make MiniMax H3 (comfy/ldm/minimax, PR #15224) numerically stable in fp16
# compute on GPUs without bf16/fp8 hardware (V100 sm_70 etc.), at near-fp16
# speed. MiniMax H3 officially supports only bf16/fp32 -- its activations
# (residual stream, gated-silu products, fc2 outputs) genuinely reach ~5e5,
# far beyond the fp16 limit (+-65504). Forcing fp16 compute => NaN => black
# frames. This plugin keeps the fp32 residual stream, but runs every big
# matmul on fp16 Tensor Cores with power-of-2 scaling compensations:
#
#   * residual stream: fp32 accumulation (_dit_block_forward)
#   * RMSNorm (210x): fp32 compute, I/O dtype preserved
#   * attention: always-fp16 SDPA via power-of-2 input scaling
#     (q/k restored by RMSNorm homogeneity, v unscaled by output multiply;
#     internal fp32 accumulate + max-subtract keep huge logits stable)
#   * MLP (v6): FULLY fp16 -- fc1 output (measured max ~585) stays fp16,
#     gated-silu scales the gate branch by a power of 2 so the product stays
#     bounded (~2340), fc2 runs fp16 with input scaling; NO fp32 intermediate
#     tensors at all (kills the ~850GB/step cast bandwidth on long seqs)
#   * long sequences: chunked MLP so activation peak stays ~9GB regardless of
#     seq length (avoids lowvram weight-eviction thrash on 16GB cards)
#   * video VAE: fp16 stream, only norms/scores/silu upcast to fp32
#
# v6.7.0: structure-clone the module tree at patch time. ModelPatcher.clone()
# only isolates dtype state; the model instance (module tree) is SHARED, so
# instance-level forward patches would leak into the cached UNETLoader output
# (deleting the node still leaves the plugin forward active). The clone tree
# shares weight parameters but has independent module objects, so every patch
# lands on the clone and cached objects always keep their native forward.
#
# v6.8.0: ZERO per-block GPU syncs (deferred fuse). Each DiTBlock used to do
# 4 syncs: 2x magnitude probes in _fp16_safe (h.abs().max().item()) + 2x
# isfinite(...).all() fuses. Measured attention/MLP inputs are <= ~2.3e2 / ~9e1
# (far inside fp16 range), so the probes are dropped and inputs are downcast
# unconditionally. The fuses now accumulate a non-finite flag on-GPU
# (isfinite().any() has no .item() => no sync) and are checked ONCE per
# forward in a MiniMaxH3Model.forward wrapper; on the (never-fires) trip the
# whole forward is re-run in _FP32_MODE for correctness. On the 8xV100 box
# this measured 5.64 -> 5.23 s/it (-7.3%); on a 16GB lowvram card the win is
# larger because a sync also stalls the CPU->GPU weight prefetch pipeline.
#
# Verified magnitudes (real model, extreme timestep):
#   max|P@V|=706  max|out_proj|~39k  max|fc1|=585  max|silu_act|=59k
#   max|fc2|=501k  -> every stage is either fp16-safe or 2-power-scaled.

import math
import copy as _copy

import torch
import comfy.model_management
from comfy.ldm.modules import attention as attn_mod

# --- defensive imports ---
try:
    import comfy.ops as _comfy_ops
except Exception:
    _comfy_ops = None
try:
    import comfy.rmsnorm as _comfy_rmsnorm
except Exception:
    _comfy_rmsnorm = None
try:
    import comfy.quant_ops as _comfy_quant_ops
except Exception:
    _comfy_quant_ops = None
try:
    import comfy.ldm.minimax.model as mm_model
    MINIMAX_AVAILABLE = True
    _IMPORT_ERR = None
except Exception as e:
    mm_model = None
    MINIMAX_AVAILABLE = False
    _IMPORT_ERR = e

ck = getattr(_comfy_quant_ops, "ck", None)
_ORIG_OPT = getattr(attn_mod, "optimized_attention", None)


class _FP32RMSNorm(torch.nn.Module):
    """RMSNorm computing in fp32, keeping I/O dtype. Exposes .weight/.eps/.bias."""
    def __init__(self, orig):
        super().__init__()
        self.orig = orig
        self.weight = orig.weight
        self.eps = orig.eps
        if hasattr(orig, "bias"):
            self.bias = orig.bias

    def forward(self, x, *args, **kwargs):
        return self.orig(x.float(), *args, **kwargs).to(x.dtype)


class _FP32LinearWrap(torch.nn.Module):
    """Linear always computing in fp32 (protects the Qwen text path)."""
    def __init__(self, orig):
        super().__init__()
        self.orig = orig

    def forward(self, x, *args, **kwargs):
        return self.orig(x.float(), *args, **kwargs)


def _is_rmsnorm(mod):
    return type(mod).__module__ == "comfy.ops" and type(mod).__name__ == "RMSNorm"


def _wrap_rmsnorms(root):
    replaced = 0
    for name, mod in list(root.named_modules()):
        if _is_rmsnorm(mod):
            parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
            parent = root.get_submodule(parent_name) if parent_name else root
            setattr(parent, attr, _FP32RMSNorm(mod))
            replaced += 1
    return replaced


def _structure_clone(module):
    """v6.7.0: 递归重建模块树 — 结构与原树完全独立, 但权重参数/缓冲区共享引用。

    ModelPatcher.clone() 只隔离 ModelPatcher 的 dtype 状态 (object_patches /
    force_cast_weights), model 实例 (模块树) 仍然共享。若直接把 forward patch
    打在共享模块上, 补丁会残留在 UNETLoader 缓存的原始对象里: 删除本节点重跑
    时缓存命中, 采样器仍走插件的 fp16 forward -> 与"未删除"行为相同, 只有
    重新加载模型 (新模块树) 才恢复原生行为。

    本函数逐节点浅拷贝、递归重建 _modules 容器, 得到一棵模块对象独立、参数
    共享的新树。所有实例级补丁 (forward / RMSNorm 包装 / condition_proj) 只
    落在这棵克隆树上; 缓存对象始终持有原生 forward。内存增量仅模块对象本身
    (数百个, 约几 MB), 时间毫秒级; 参数共享保证 ComfyUI 权重 cast 与
    lowvram 换入换出照常工作。
    """
    new = _copy.copy(module)
    new._modules = dict(module._modules)
    for name, child in module._modules.items():
        if child is not None:
            new._modules[name] = _structure_clone(child)
    return new


# ---- v6.8.0: ZERO per-block GPU syncs (deferred fuse) ----
# Old per-DiTBlock cost: 4 syncs (2x _fp16_safe magnitude probes + 2x
# isfinite().all() fuses, each a GPU->CPU round trip). On a lowvram card a
# sync also stalls the CPU->GPU weight prefetch pipeline, so it is MORE
# expensive than on a GPU-resident box (dg1kjd measured 7.3% on 8xV100-32GB).
# v6.8.0 drops the probes (inputs measured <= ~2.3e2 attn / ~9e1 MLP) and
# accumulates the isfinite fuses on-GPU (no .item()); MiniMaxH3Model.forward
# is wrapped to check the flag ONCE per forward and, if it ever trips,
# re-runs the whole forward in _FP32_MODE (correct-by-construction slow path).
_FP32_MODE = False              # set during the rare fp32 re-run
_FUSE_FLAG = None               # lazy GPU bool tensor (accumulated, never sync'd per block)


def _accum_fuse(t):
    """On-GPU non-finite accumulation. No .item() => no device sync."""
    global _FUSE_FLAG
    if _FUSE_FLAG is None or _FUSE_FLAG.device != t.device:
        _FUSE_FLAG = torch.zeros((), device=t.device, dtype=torch.bool)
    _FUSE_FLAG |= (~torch.isfinite(t)).any()


def _model_fwd_wrapper(orig_forward):
    """Wrap MiniMaxH3Model.forward: check the deferred fuse once per forward;
    on a trip (mathematically impossible with the fixed scales) re-run in fp32."""
    def wrapper(self, *args, **kwargs):
        global _FP32_MODE, _FUSE_FLAG
        if _FUSE_FLAG is not None:
            _FUSE_FLAG.zero_()
        _FP32_MODE = False
        out = orig_forward(*args, **kwargs)
        if _FUSE_FLAG is not None and bool(_FUSE_FLAG.item()):
            print("[MiniMaxH3-FP16Safe] WARNING: fuse tripped (unexpected non-finite); "
                  "re-running forward in fp32 mode")
            _FP32_MODE = True
            _FUSE_FLAG.zero_()
            out = orig_forward(*args, **kwargs)
            _FP32_MODE = False
        return out
    return wrapper


def _fp16_downcast(h):
    """v6.8.0: unconditional fp16 downcast (no .item() probe). Inputs measured
    <= ~2.3e2 (attn) / ~9e1 (MLP), far inside fp16 range; a theoretical >65504
    input becomes inf and is caught by the deferred fuse -> fp32 re-run."""
    if h.dtype == torch.float16:
        return h
    if _FP32_MODE:
        return h
    return h.to(torch.float16)


# legacy alias kept for clarity in older call sites; identical to _fp16_downcast
def _fp16_safe(h):
    return _fp16_downcast(h)


def _fp16_scaled(h, threshold=60000.0):
    """V3: downcast to fp16 ALWAYS, scaling by a power of 2 when needed.

    Returns (fp16_tensor, scale) with scale = 2**k (k>=0). When |h| exceeds the
    fp16-safe threshold, h is divided by scale before the cast, so the fp16
    matmuls never overflow even for huge residuals. The scale is a power of two,
    so the division is exact (no precision loss); the caller compensates:
      * qkv_proj is linear -> q' = q / scale, and RMSNorm(q') == RMSNorm(q)
        (normalization cancels the scaling), so q/k are restored for free;
      * v does NOT pass RMSNorm -> the SDPA output must be multiplied back by scale.
    """
    if h.dtype == torch.float16:
        return h, 1.0
    try:
        m = h.abs().max().item()
        if m <= threshold:
            return h.to(torch.float16), 1.0
        k = int(math.ceil(math.log2(m / threshold)))
        s = float(1 << k)
        return (h / s).to(torch.float16), s
    except Exception:
        pass
    return h, 1.0  # caller must fall back to fp32 when dtype is still fp32


def _qkv_scale_check(self, x, x_scale):
    """V4: qkv_proj output can exceed fp16 range even when its input h is small
    (the linear layer amplifies by the weight norm). Detect an fp16 overflow of
    q/k/v and redo qkv_proj with a larger power-of-2 scale on x.
    V4: single .item() (v3 did 3) -- q/k/v share one qkv tensor, so one abs-max
    over the full qkv covers all three. Halves the per-layer GPU sync count.
    Returns (q, k, v in fp16, final x_scale)."""
    qkv = self.qkv_proj(x)
    try:
        if qkv.abs().max().item() > 60000.0 and x_scale < 1e6:
            qkv_max = qkv.abs().max().item()
            extra = int(math.ceil(math.log2(qkv_max / 60000.0)))
            x_scale2 = x_scale * float(1 << extra)
            xh2 = (x.float() / x_scale2).to(torch.float16)
            qkv = self.qkv_proj(xh2)
            x_scale = x_scale2
    except Exception:
        pass
    q, k, v = qkv.split(self.heads * self.head_dim, dim=-1)
    return q, k, v, x_scale


# ---- DiT Attention (v6.3+): ALWAYS-fp16 SDPA with FIXED power-of-2 scale ----
# Zero per-layer scans. x (fp16 or fp32) is always divided by the scale below
# (an exact power of 2) before qkv_proj, so qkv output is bounded by
# |x|/s * ||W_qkv||. q/k are restored by RMSNorm homogeneity; v stays scaled
# through the linear chain SDPA -> out_proj and is unscaled in fp32 afterwards.
#
# The scale is bounded BELOW by overflow and ABOVE by accuracy:
#   * overflow: the binding site is out_proj's output, the largest fp16 tensor
#     in this path (~39k at 480p; 1.43e5 measured at 1344x768/124f). s=16 leaves
#     ~27x headroom on the former, 7.3x on the latter, with the isfinite fuse
#     behind it.
#   * accuracy: RMSNorm is scale-homogeneous only as eps -> 0, because
#     rms_norm(q/s) = q / sqrt(ms_q + eps*s^2). With qk_norm_eps=1e-5, s=256
#     makes that term 0.655, which is NOT small next to the live ms(q)
#     (measured median ~9-10, min ~2.2 at block 0). Attention-output error vs
#     an unprescaled fp32 reference on identical activations follows 1/s^2
#     exactly, with no plateau (verified on real model, 52 layers):
#         /256 mean rel 1.35e-2        /32  mean rel 2.25e-4
#         /64  mean rel 8.99e-4        /16  mean rel 5.62e-5  (~240x better)
_ATTN_FIXED_SCALE = 16.0   # 2^4 (v6.6.0, was 256.0; PR #1)


def _dit_attn_forward(self, x, rope_freqs=None, transformer_options={}):
    s = x.shape[0]
    if _FP32_MODE:
        x_h = x * (1.0 / _ATTN_FIXED_SCALE)              # fp32 re-run: keep fp32
    elif x.dtype == torch.float16:
        x_h = x * (1.0 / _ATTN_FIXED_SCALE)              # exact in fp16
    else:
        x_h = (x * (1.0 / _ATTN_FIXED_SCALE)).to(torch.float16)
    qkv = self.qkv_proj(x_h)
    q, k, v = qkv.split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if ck is not None:
            if comfy.model_management.in_training:
                q, k = ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            else:
                ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            q = self.q_norm(q[0]).unsqueeze(0)
            k = self.k_norm(k[0]).unsqueeze(0)
            print("[MiniMaxH3-FP16Safe] WARNING: comfy_kitchen missing, rope skipped.")
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
    # [1, heads, s, hd]; q/k restored to O(1) by RMSNorm, v still /16
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    if q.dtype == torch.float16:
        # 走 ComfyUI optimized_attention 分发：transformer_options 含 override
        # （如 H3_V100 的 V100 FlashAttention）时由 override 接管（FA2 返回 [B,S,H*D]）；
        # 无 override 时 pytorch 后端 = 直调 SDPA（返回 [B,H,S,D]），行为不变。
        out = attn_mod.optimized_attention(
            q, k, v, self.heads, mask=None, skip_reshape=True,
            transformer_options=transformer_options)
        out = out.transpose(1, 2).reshape(s, -1) if out.dim() == 4 else out.reshape(s, -1)
    else:
        out = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
        out = out.transpose(1, 2).reshape(s, -1)                          # [s, heads*hd]
    proj = self.out_proj(out)                                             # fp16, bounded <= ~2400
    proj = proj.float() * _ATTN_FIXED_SCALE                               # unscale v in fp32
    if not _FP32_MODE:
        _accum_fuse(proj)                                                 # v6.8.0: deferred, no sync
    return proj                                                           # fp32 for the stream


# ---- DiT block / refiner block: fp32 residual stream, fp16 inner compute ----
import os as _os
import time as _time
_ENV_PROFILE = _os.environ.get("MINIMAXH3_PROFILE", "").strip().lower() in ("1", "true", "yes", "on")
_PROFILE = _ENV_PROFILE          # 进程内当前开关状态, 每次 patch 重新按 env/节点开关决定
_prof = {"attn": 0.0, "mlp": 0.0, "other": 0.0, "n": 0}


def _prof_block(i, kind, dt_attn, dt_mlp, dt_other):
    _prof["attn"] += dt_attn
    _prof["mlp"] += dt_mlp
    _prof["other"] += dt_other
    _prof["n"] += 1
    idx = i if i >= 0 else _prof["n"]        # fallback: call counter
    if idx in (2, 5, 10, 25, 50, 100):
        alloc = torch.cuda.memory_allocated() / 2**30
        print(f"[MiniMaxH3-FP16Safe][PROF] block {idx} ({kind}): attn累计={_prof['attn']:.1f}s "
              f"mlp累计={_prof['mlp']:.1f}s other累计={_prof['other']:.1f}s "
              f"当前allocated={alloc:.1f}GB", flush=True)


def _dit_block_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
    _t = _time.time()
    _mss = mm_model._mod_scale_shift
    _mg = mm_model._mod_gate
    x = x.float()
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
    h = _mss(self.norm1(x), shift_msa, scale_msa, mod_segments)
    _t1 = _time.time()
    # V4: _fp16_safe pre-check kept -- it downcasts safe residuals to fp16 here
    # so q_norm/k_norm (_FP32RMSNorm) output stays fp16 and SDPA stays on the
    # fast fp16 path; removing it made norm output fp32 -> SDPA fp32 -> NaN.
    attn_out = self.attn(_fp16_safe(h), rope_freqs=rope_freqs,
                         transformer_options=transformer_options).float()
    _t2 = _time.time()
    x = _mg(x, gate_msa, attn_out, mod_segments)
    h = _mss(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
    mlp_out = self.mlp(_fp16_safe(h)).float()
    _t3 = _time.time()
    out = _mg(x, gate_mlp, mlp_out, mod_segments)
    _t4 = _time.time()
    if _PROFILE:
        idx = getattr(self, "_dbg_index", -1)
        _prof_block(idx, "dit", _t2 - _t1, _t3 - _t2, (_t1 - _t) + (_t4 - _t3))
    return out


def _refiner_block_forward(self, x, transformer_options={}):
    x = x.float()
    a = self.attn(_fp16_safe(self.norm1(x)), transformer_options=transformer_options).float()
    x = x.add_(a)
    m = self.mlp(_fp16_safe(self.norm2(x))).float()
    return x.add_(m)


# ---- DiT MLP (v6): fully-fp16 MLP with power-of-2 gate scaling ----
# Measured magnitudes (real model, extreme timestep): max|fc1|=585 (fp16-safe),
# max|silu_act|=59k (close to fp16 limit!), max|fc2|=501k (~8.5x input).
# v5 chunking fixed the VRAM cliff (110->97s) but the per-layer fp32 cast of the
# fc1 output (5.6GB fp16 -> 11.3GB fp32, x50 layers/step) is ~30s of pure extra
# bandwidth. v6 keeps the WHOLE MLP in fp16:
#   * fc1 out (max ~585) stays fp16;
#   * gated-silu: scale the gate branch b by a power of 2 so that
#     act = silu(a) * (b/2^k) stays well inside fp16 (target act <= ~585*4);
#   * fc2 runs fp16; its output ~8.5x input, so scale the act input when needed;
#     unscale everything in fp32 at the end (small [s,3072] tensor).
#   * when the residual was scaled (x_scale != 1, rare) silu must see true
#     values -> fall back to the chunked-fp32 path (correct, slow, rare).
_MLP_CHUNK = 16384            # rows per chunk


def _mlp_chunked_fp32(self, x_h, x_scale, s):
    """Correct-but-slow fp32 chunked path (used when x was power-of-2 scaled)."""
    outs = []
    for i in range(0, s, _MLP_CHUNK):
        xc = x_h[i:i + _MLP_CHUNK]
        y = self.fc1(xc).float()
        if x_scale != 1.0:
            y.mul_(x_scale)
        a, b = y.chunk(2, dim=-1)
        act = torch.nn.functional.silu(a)
        act.mul_(b)
        del y
        try:
            amax = act.abs().max().item()
        except Exception:
            amax = 0.0
        if amax > 4000.0:
            sc = 1 << int(math.ceil(math.log2(max(amax / 4000.0, 1.0))))
            outs.append(self.fc2((act / sc).to(torch.float16)).float() * sc)
        else:
            outs.append(self.fc2(act.to(torch.float16)).float())
    return torch.cat(outs, dim=0)


def _mlp_forward(self, x):
    s = x.shape[0]
    if _FP32_MODE or x.dtype != torch.float16:
        # fp32 re-run (fuse trip, mathematically impossible): full fp32 path.
        # comfy.ops casts weights to the input dtype, so fp32 in -> fp32 matmul.
        return _mlp_chunked_fp32(self, x.float(), 1.0, s)
    x_h = x  # already fp16 (v6.8.0: downcast happened upstream, no probe)
    # fast path: fully fp16 MLP, ZERO per-chunk syncs (v6.2).
    # Fixed conservative scales (no .item() scans):
    #   * bs=16 on the gate branch: measured max|fc1|=585 -> act = silu(a)*(b/16)
    #     <= 585*36.6 = 21.4k (fp16-safe), with ~3.4x headroom for outliers;
    #   * fs=8 on the act input: fc2 out ~8.5x input -> <= 21.4k/8*8.5 = 22.7k.
    # Both are powers of 2 (exact in fp16); unscale in fp32 at the end. The
    # deferred isfinite fuse catches any unexpected overflow (rare fp32 re-run).
    _BS = 16.0
    _FS = 8.0
    outs = []
    for i in range(0, s, _MLP_CHUNK):
        xc = x_h[i:i + _MLP_CHUNK]
        y16 = self.fc1(xc)                       # fp16 [c, 2I], max ~585 (safe)
        a, b = y16.chunk(2, dim=-1)
        act = torch.nn.functional.silu(a) * (b * (1.0 / _BS))        # fp16 gated-silu
        outs.append(self.fc2(act * (1.0 / _FS)).float() * _FS * _BS) # fp16 fc2, fp32 unscale
    out = torch.cat(outs, dim=0)
    _accum_fuse(out)                             # v6.8.0: deferred, no sync
    return out


# ---- Video VAE (v7): 不再 patch VAE ----
# 2026-08-09 实测: VAE 原生 fp16 解码 finite=True (含 outlier ±38), 与 fp32 保险路径
# 差异仅 0.5%, 且快 ~30%; 原生 Attention 自带 nan_to_num 保险。故插件完全不管 VAE,
# 节点也不再有 vae 输入/输出端口。

def _is_minimax_dit(inner):
    if inner is None or not hasattr(inner, "modules"):
        return False
    if any(type(m).__name__ == "MiniMaxH3Model" for m in inner.modules()):
        return True
    for m in inner.modules():
        if all(hasattr(m, a) for a in ("qkv_proj", "out_proj", "q_norm", "k_norm")):
            return True
    return False


class MiniMaxH3FP16Safe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {
                "debug_nan": ("BOOLEAN", {"default": False}),
                "profile": ("BOOLEAN", {"default": False, "tooltip": "打印每阶段耗时(block 2/5/10/25/50)与显存, 定位速度瓶颈"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "MiniMaxH3"

    def patch(self, model, debug_nan=False, profile=False):
        if not MINIMAX_AVAILABLE:
            print("[MiniMaxH3-FP16Safe] comfy.ldm.minimax backend not found (%s). "
                  "Update ComfyUI to a build that includes PR #15224." % _IMPORT_ERR)
            return (model,)

        # ---- v6.6.0 (Issue #2 fix): 在 clone 上 patch, 隔离 ModelPatcher 持久状态 ----
        # set_model_compute_dtype 会把 manual_cast_dtype + force_cast_weights 写入
        # ModelPatcher.model_options/属性 (持久状态), 而 UNETLoader 输出被 ComfyUI 缓存。
        # 若直接在缓存对象上修改, 删除本节点后重跑会残留 fp16 compute 却无缩放补偿
        # (forward patch 是实例级、随新实例消失) -> 回到 NaN/黑帧条件。
        # clone() 深拷贝 model_options/object_patches, 使缓存中的原始对象保持干净;
        # 模型实例共享, 但 comfy.ops 的权重自动 cast 保证 fp16 输入下计算正确。
        if not getattr(model, "_h3fp16safe_patched", False):
            try:
                model = model.clone()
                model._h3fp16safe_patched = True
                print("[MiniMaxH3-FP16Safe] patching on a cloned ModelPatcher "
                      "(cache object left untouched, Issue #2)")
            except Exception as e:
                print("[MiniMaxH3-FP16Safe] WARNING: model.clone() failed (%s); "
                      "patching in place (cache may retain fp16 compute dtype)." % e)

        # ---- v6.7.0 (Issue #2 forward side): 结构克隆模块树 ----
        # ModelPatcher.clone() 只隔离 dtype 状态, 模块树仍与原对象共享; 直接把
        # forward patch 打在共享模块上会残留在缓存的 UNETLoader 输出里 (删除
        # 本节点重跑仍是插件行为)。这里重建一棵结构独立、参数共享的模块树,
        # 后续所有实例级补丁只落在克隆树上, 缓存对象永远保持原生 forward。
        inner = getattr(model, "model", None)
        if inner is not None:
            try:
                model.model = _structure_clone(inner)
                print("[MiniMaxH3-FP16Safe] structure-cloned model tree "
                      "(params shared, module instances isolated)")
            except Exception as e:
                print("[MiniMaxH3-FP16Safe] WARNING: structure clone failed (%s); "
                      "patching shared tree (cache may retain forward patches)." % e)

        # ---- 强制 fp16 计算: MiniMaxH3 官方 supported=[bf16,fp32] 不含 fp16,
        # V100 无 bf16 硬件 -> aki 兜底 cast 成 fp32 (慢 4x, 无 Tensor Core)。
        # 必须在 patch 时强制 fp16, 否则 qkv/out/fc1/fc2 全是 fp32 matmul。
        try:
            model.set_model_compute_dtype(torch.float16)
            print("[MiniMaxH3-FP16Safe] forced compute dtype -> fp16 (weights cast to fp16, Tensor Core ON)")
        except Exception as e:
            print("[MiniMaxH3-FP16Safe] WARNING: set_model_compute_dtype(fp16) failed: %s" % e)

        # ---- DiT (UNet): 实例级 forward patch (不再改类方法, 避免全局生效) ----
        # 只替换当前 model 实例内模块的 forward; 之后新加载的其他 MiniMax 模型
        # (工作流中无本节点) 保持原生 forward。V100 的 fp16 修正只作用于本实例。
        import types as _types
        global _PROFILE, _prof
        # 节点开关可真正关闭; 只有环境变量显式开启时 profile 才强制打开
        _PROFILE = _ENV_PROFILE or bool(profile)
        _prof = {"attn": 0.0, "mlp": 0.0, "other": 0.0, "n": 0}   # 每次 patch 重置累计, 避免跨多次运行叠加
        inner = getattr(model, "model", None)
        if _is_minimax_dit(inner):
            target = getattr(inner, "diffusion_model", None) or inner
            patched = 0
            for m in target.modules():
                if isinstance(m, mm_model.DiTBlock):
                    m.forward = _types.MethodType(_dit_block_forward, m)
                    patched += 1
                elif isinstance(m, mm_model.RefinerBlock):
                    m.forward = _types.MethodType(_refiner_block_forward, m)
                    patched += 1
                elif isinstance(m, mm_model.MLP):
                    m.forward = _types.MethodType(_mlp_forward, m)
                    patched += 1
                elif isinstance(m, mm_model.Attention):
                    m.forward = _types.MethodType(_dit_attn_forward, m)
                    patched += 1
            n = _wrap_rmsnorms(target)
            wrapped_cond = 0
            for m in target.modules():
                if hasattr(m, "condition_proj") and isinstance(m.condition_proj, torch.nn.Module):
                    m.condition_proj = _FP32LinearWrap(m.condition_proj)
                    wrapped_cond += 1
            # 总是给 block 打索引 (debug_nan 与 profile 都依赖它)
            fwd_wrapped = 0
            try:
                for m in target.modules():
                    if isinstance(m, mm_model.MiniMaxH3Model):
                        for i, blk in enumerate(m.blocks):
                            blk._dbg_index = i
                        # v6.8.0: 包装根 forward, deferred fuse 每 forward 检查一次
                        if not getattr(m, "_h3fp16safe_fwd_wrapped", False):
                            m.forward = _types.MethodType(_model_fwd_wrapper(m.forward), m)
                            m._h3fp16safe_fwd_wrapped = True
                            fwd_wrapped = 1
                        break
            except Exception:
                pass
            global _FUSE_FLAG, _FP32_MODE
            _FUSE_FLAG = None
            _FP32_MODE = False
            print("[MiniMaxH3-FP16Safe][V6.8-NOSYNC] DiT patched (instance-level, %d modules): "
                  "fp32 residual stream + fp16 SDPA attention (fixed /16 scale, zero scans) + "
                  "fully-fp16 MLP (fixed-scale) + %d RMSNorm(s) + %d condition_proj. "
                  "zero per-block GPU syncs (deferred fuse, fwd_wrapped=%d). "
                  "(profile=%s)" % (patched, n, wrapped_cond, fwd_wrapped, _PROFILE))

            if debug_nan:
                self._install_nan_debug(mm_model, target)
        else:
            print("[MiniMaxH3-FP16Safe] MODEL is not MiniMax H3; DiT left unchanged.")

        return (model,)

    @staticmethod
    def _install_nan_debug(mm_model, target=None):
        """实例级 NaN 检测: 包装每个 DiTBlock 实例的 forward (基于实例当前 forward,
        即已安装的 fp16 patch), 不再做类级替换, 不影响其他模型实例。"""
        import types as _types
        if target is None:
            print("[MiniMaxH3-FP16Safe] debug_nan: no target, skipped.")
            return
        try:
            blocks = None
            for m in target.modules():
                if isinstance(m, mm_model.MiniMaxH3Model):
                    blocks = m.blocks
                    break
        except Exception:
            blocks = None
        if not blocks:
            print("[MiniMaxH3-FP16Safe] debug_nan: no MiniMaxH3Model found, skipped.")
            return
        state = {"reported": False}
        for blk in blocks:
            idx = getattr(blk, "_dbg_index", -1)
            base = blk.forward  # 实例当前 forward (可能已是 fp16 patch)

            def checking(self, x, t_emb, mod_segments, rope_freqs, transformer_options={},
                         _base=base, _idx=idx, _state=state):
                if not _state["reported"] and not torch.isfinite(x).all():
                    print("[MiniMaxH3-FP16Safe][DEBUG] DiTBlock %s INPUT already non-finite "
                          "(finite %.4f) -> NaN originates BEFORE the blocks "
                          "(embedding/refiner/context)" % (
                              _idx, torch.isfinite(x).float().mean().item()))
                    _state["reported"] = True
                out = _base(x, t_emb, mod_segments, rope_freqs, transformer_options)
                if not _state["reported"] and not torch.isfinite(out).all():
                    print("[MiniMaxH3-FP16Safe][DEBUG] non-finite value first seen at DiTBlock %s "
                          "(finite ratio %.4f)" % (_idx, torch.isfinite(out).float().mean().item()))
                    _state["reported"] = True
                return out

            blk.forward = _types.MethodType(checking, blk)
        print("[MiniMaxH3-FP16Safe] debug_nan enabled (instance-level, %d blocks)." % len(blocks))

