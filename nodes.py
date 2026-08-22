# -*- coding: utf-8 -*-
"""Sol-Attn (V100) — MiniMax H3 在 V100 上的注意力稀疏加速（ComfyUI 自定义节点）。

单节点 = 内嵌 FP16Safe（fp16 安全：prescale /16 + 熔断 + fp32 重跑兜底）
        + Sol-Attn 官方稀疏路由（keep-or-drop kernel，预编译 pyd）。

引用与使用的内容（详见 README/DEVELOPMENT）：
- Sol-Attn（arXiv 2607.24027，NVlabs/Sana sol-engine）：kc/vc 块统计 + diag 阈值 +
  列均值路由 + keep-or-drop —— 稀疏算法与路由（routing.py 按官方 preprocess.py 复刻）
- keep-or-drop kernel：flash-attn 官方 sparse kernel（Tri Dao）的 sm70 适配版，
  预编译 pyd 分发（comfy_v100_solattn_cuda.cp312-win_amd64.pyd）
- FP16Safe（ComfyUI-MiniMaxH3-FP16Safe v6.8.0）：fp16 NaN 安全逻辑内嵌为 fp16safe.py
  （prescale /16 + 熔断 + fp32 重跑兜底，自包含无需单独安装）

v1.0.0（2026-08-20 正式版）：dense 兜底 = 原版 SDPA；参数对齐 kijai/
ComfyUI-SolAttn_triton 风格（tau / start_percent / end_percent / min_tokens /
dense_blocks / h3_prefix_tokens）；sparse-only kernel（native/ 源码可编译）。
"""
import logging
import math
from pathlib import Path

import torch

LOGGER = logging.getLogger("SolAttn")
SPARSE_KEY = "solattn_sparse"
TAU_KEY = "solattn_tau"
START_PERCENT_KEY = "solattn_start_percent"
END_PERCENT_KEY = "solattn_end_percent"
MIN_TOKENS_KEY = "solattn_min_tokens"
DENSE_BLOCKS_KEY = "solattn_dense_blocks"
PREFIX_TOKENS_KEY = "solattn_prefix_tokens"
TOPK_KEY = "solattn_topk"
_BLOCK_INDEX_HOOKED = set()
_EXTENSION_LOADED = False
_REPORTED_STATS = set()

_PYD_NAME = "comfy_v100_solattn_cuda.cp312-win_amd64.pyd"


def _load_extension():
    """加载预编译 kernel pyd（含 keep-or-drop sparse op；随插件分发，无需本地编译）。

    按名称前缀 ``comfy_v100_solattn_cuda*.pyd`` 匹配：预编译版（cp312）与其他用户
    自行编译的版本（cp311/cp313 等，文件名后缀随 Python 版本变化）均可识别。
    """
    global _EXTENSION_LOADED
    if _EXTENSION_LOADED:
        return
    pyd_dir = Path(__file__).resolve().parent
    candidates = sorted(pyd_dir.glob("comfy_v100_solattn_cuda*.pyd"))
    if not candidates:
        raise RuntimeError(
            f"SolAttn: 缺少预编译 kernel {_PYD_NAME}（应随插件分发在插件目录下，"
            f"或从 Release 下载 / 自行编译 native/ 后放入插件目录）"
        )
    torch.ops.load_library(str(candidates[0]))
    _EXTENSION_LOADED = True


def _load_fp16safe_nodes():
    """返回内嵌的 FP16Safe 实现模块（fp16safe.py，自包含，无需单独安装 FP16Safe 插件）。"""
    from . import fp16safe
    return fp16safe


class _quiet_print:
    """临时抑制 print 输出（内嵌 FP16Safe 的 [MiniMaxH3-FP16Safe] 噪声）。

    patch 发生在工作流加载阶段（单线程），短暂替换 builtins.print 后恢复。
    """

    def __enter__(self):
        import builtins
        self._orig = builtins.print
        builtins.print = lambda *a, **k: None

    def __exit__(self, *exc):
        import builtins
        builtins.print = self._orig
        return False


def parse_blocks(spec, count):
    """解析块规格为绝对索引集合（kijai 同款语法）。

    "0-1" -> {0,1}；"0-2,-1" -> {0,1,2,count-1}；空 -> 空集合。
    负数为从末尾倒数（-1 = 最后一层）。
    """
    if not spec:
        return frozenset()
    out = set()
    for part in spec.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        if part.startswith("-") and part.count("-") == 1:
            out.add(int(part) + count)                 # 单个负数 "-1"
        elif "-" in part:
            a, b = part.split("-", 1)                  # 区间 "0-2"
            lo, hi = int(a), int(b)
            if lo < 0:
                lo += count
            if hi < 0:
                hi += count
            out.update(range(lo, hi + 1))
        else:
            idx = int(part)
            if idx < 0:
                idx += count
            out.add(idx)
    return frozenset(out)


# ---------------- 稀疏路径 ----------------
def _sparse_attn(q, k, v, heads, scale, opts):
    """稀疏路径: Sol-Attn 官方路由 + keep-or-drop sparse kernel。q,k,v: (B,H,S,D) fp16

    优化：路由用连续 q/k（非连续 einsum 慢）；kernel 直接用非连续
    q/k/v（kernel 按 stride 访问，实测非连续更快，省 3×contiguous 拷贝）。
    """
    from . import routing
    batch = q.shape[0]
    if batch != 1:
        return None
    q3 = q.squeeze(0).permute(1, 0, 2)                    # [S,H,D] 非连续（kernel 支持）
    k3 = k.squeeze(0).permute(1, 0, 2)
    v3 = v.squeeze(0).permute(1, 0, 2)
    S, H, D = q3.shape
    if S < 64 or H != heads:
        return None                                       # 过短/形状不符
    if not hasattr(torch.ops.comfy_v100_solattn_cuda, "varlen_fwd_sparse"):
        return None                                       # pyd 未含 sparse op（异常分发）
    tau = float(opts.get(TAU_KEY, 1.0))
    prefix_tokens = int(opts.get(PREFIX_TOKENS_KEY, 0))
    topk_k = int(opts.get(TOPK_KEY, 0))
    # v1.1: routing 直接吃非连续 q3/k3（内部视图化块统计，零 pad 拷贝）；
    # nnz_s 缩小 off 到 NB/2 槽（kernel 运行时读 num_blks，多余槽位不读，省 ~50% 路由显存）。
    nnz_s = max(256, (S + 63) // 64 // 2)
    cnt, off, ccnt, cidx = routing.build_sparse_csr(
        q3, k3,
        tau=tau, scale=scale, sink_tokens=prefix_tokens, nnz_s=nnz_s, topk_k=topk_k)
    # 首次调用打印序列统计（prefix debug：确认 sink_tokens 是否覆盖 text+cond+ref+audio）
    if S not in _REPORTED_STATS:
        _REPORTED_STATS.add(S)
        dens = float(cnt.float().mean().item()) / max((S + 63) // 64, 1)
        print(f"[SolAttn][v1.1.1] S={S} blocks={math.ceil(S / 64)} 密度≈{dens * 100:.1f}% "
              f"| prefix_tokens={prefix_tokens}（建议 = text+cond+ref+audio 实际 token 数，"
              f"用于 sink 保底）| topk_blocks={topk_k}", flush=True)
    softmax_scale = float(scale if scale is not None else 1.0 / math.sqrt(D))
    cu = torch.tensor([0, S], dtype=torch.int32, device=q.device)
    out, _lse = torch.ops.comfy_v100_solattn_cuda.varlen_fwd_sparse(
        q3, k3, v3, None, cu, cu, cnt, off, ccnt, cidx, S, S, softmax_scale)
    return out.reshape(1, S, H * D)                       # [B,S,H*D]（与 comfy_attention 同布局）


def _unsupported_reasons(q, k, v, heads, mask, skip_reshape, skip_output_reshape, opts):
    """返回不走稀疏的原因列表（非空 = 该 attention 回退原版 SDPA）。"""
    reasons = []
    if mask is not None:
        reasons.append("mask")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        reasons.append("dtype")
    S = q.shape[-2] if q.dim() >= 3 else 0
    min_tokens = int(opts.get(MIN_TOKENS_KEY, 1024))
    if S < min_tokens:
        reasons.append("min-tokens")
    # 采样窗口（kijai 语义：percent 窗口内稀疏，窗口外 dense）
    start_percent = float(opts.get(START_PERCENT_KEY, 0.2))
    end_percent = float(opts.get(END_PERCENT_KEY, 1.0))
    step = opts.get("step")
    total = opts.get("total_steps")
    if step is not None and total:
        frac = int(step) / max(int(total) - 1, 1)
        if frac < start_percent or frac > end_percent:
            reasons.append("sampling-window")
    else:
        sigmas = opts.get("sigmas")
        if sigmas is not None and len(sigmas) > 0 and float(sigmas[0]) > 14.0:
            reasons.append("sigma-warmup")               # turbo 4-step 第一步 sigma≈14.64
    # dense_blocks（用户指定的保留 dense 层）
    dense_blocks = opts.get(DENSE_BLOCKS_KEY) or frozenset()
    block = opts.get("solattn_block")
    if block is not None and block in dense_blocks:
        reasons.append("dense-block")
    return reasons


def _install_block_index(model):
    """发布当前 block index 到 transformer_options（kijai 同款机制，供 dense_blocks 判断）。"""
    blocks = getattr(model, "blocks", None)
    if blocks is None or id(model) in _BLOCK_INDEX_HOOKED:
        return
    for index, block in enumerate(blocks):
        def make_hook(index):
            def hook(_module, _args, kwargs):
                options = kwargs.get("transformer_options")
                if isinstance(options, dict):
                    options["solattn_block"] = index
                return None
            return hook
        block.register_forward_pre_hook(make_hook(index), with_kwargs=True)
    _BLOCK_INDEX_HOOKED.add(id(model))


def solattn_attention_override(original, q, k, v, heads, mask=None, attn_precision=None,
                               skip_reshape=False, skip_output_reshape=False, **kwargs):
    """attention override: 满足条件走 Sol-Attn 稀疏，否则原版 SDPA 兜底。"""
    opts = kwargs.get("transformer_options") or {}
    if opts.get(SPARSE_KEY, False):
        reasons = _unsupported_reasons(q, k, v, heads, mask, skip_reshape,
                                       skip_output_reshape, opts)
        if not reasons:
            try:
                out_s = _sparse_attn(q, k, v, heads, kwargs.get("scale"), opts)
                if out_s is not None:
                    return out_s
            except Exception as exc:
                LOGGER.warning("[SolAttn] sparse fallback: %s", exc)
    # ---- dense 兜底 = 原版 SDPA（无 FA2，绕开高显存峰值）----
    return original(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                    skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
                    **kwargs)


solattn_attention_override._solattn_override = True


# ---------------- 节点 ----------------
class SolAttnV100:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "fp16_safe": ("BOOLEAN", {"default": True,
                                          "tooltip": "内嵌 FP16Safe（prescale /16 + 熔断 + fp32 重跑兜底）。关闭则需工作流中另有节点提供 fp16 安全"}),
                "tau": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 4.0, "step": 0.05,
                                  "tooltip": "稀疏路由阈值 β：越大越稀疏越快。0.75 ≈ 40% 密度（V100 真机质量≈dense，推荐默认）；1.0 ≈ 26% 密度（更快，高动态/文字场景质量下降）"}),
                "start_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "采样进度低于此比例走 dense（论文用 0.2；turbo 4-step 下第一步自然 dense）"}),
                "end_percent": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "采样进度高于此比例走 dense（0.9=尾部 10% 步保真，推荐默认；1.0=不启用尾部 dense，最快）"}),
                "min_tokens": ("INT", {"default": 1024, "min": 64, "max": 65536, "step": 64,
                                       "tooltip": "序列短于此 token 数不走稀疏（直接 SDPA）"}),
                "dense_blocks": ("STRING", {"default": "0-1,-1",
                                            "tooltip": "保留 dense 的 transformer 块，如 '0-1'=前两层，'0-2,-1'=前三层+最后一层（-1 从末尾数）；空=全部稀疏"}),
                "h3_prefix_tokens": ("INT", {"default": 1024, "min": 0, "max": 65536, "step": 64,
                                             "tooltip": "H3 序列 text/cond/ref/audio 前缀 token 数（KV sink 保底；首次运行看控制台 [SolAttn] S=... 建议 ≥ 实际前缀；图生视频参考图场景 1024 起步）"}),
                "topk_blocks": ("INT", {"default": 32, "min": 0, "max": 512, "step": 4,
                                        "tooltip": "每行保底块数（质量修复）：阈值路由是'均值对齐检测'，高动态/新内容块对齐度低会被过滤（手/边缘/肢体动态帧丢失）。topk 强制每行保留分数最高 K 块，动态内容保底。0=关闭。32 对 1540 块 ≈ 2% 密度开销"}),
            },
            "optional": {
                "debug_nan": ("BOOLEAN", {"default": False}),
                "profile": ("BOOLEAN", {"default": False, "tooltip": "打印每阶段耗时与显存（FP16Safe 侧）"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "sol_attn"

    DESCRIPTION = (
        "Sol-Attn (V100): MiniMax H3 注意力稀疏加速。单节点 = 内嵌 FP16Safe "
        "(prescale /16 + deferred fuse, fp16 安全) + Sol-Attn 官方稀疏 "
        "(keep-or-drop kernel)。480p/10s 实测 43s/步（vs 纯 FP16Safe 71-74s，~1.7×），"
        "画质肉眼无损。参数对齐 kijai/ComfyUI-SolAttn_triton 风格。"
    )

    def patch(self, model, fp16_safe=True, tau=0.75, start_percent=0.2, end_percent=0.9,
              min_tokens=1024, dense_blocks="0-1,-1", h3_prefix_tokens=1024, topk_blocks=32,
              debug_nan=False, profile=False):
        if not torch.cuda.is_available() or not any(
            torch.cuda.get_device_capability(index) == (7, 0)
            for index in range(torch.cuda.device_count())
        ):
            raise RuntimeError("SolAttn (V100) requires an SM70/V100 CUDA device.")
        # ---- 内嵌 FP16Safe: fp16 安全（prescale /16 + 熔断），自包含无需外部插件 ----
        if fp16_safe:
            fp16safe_mod = _load_fp16safe_nodes()
            with _quiet_print():
                model, = fp16safe_mod.MiniMaxH3FP16Safe().patch(
                    model, debug_nan=bool(debug_nan), profile=bool(profile)
                )
            LOGGER.info("[SolAttn] FP16Safe embedded (prescale /16 + deferred fuse)")
        _load_extension()
        patched = model.clone()
        transformer_options = patched.model_options.setdefault("transformer_options", {})
        existing = transformer_options.get("optimized_attention_override")
        if existing is not None and not getattr(existing, "_solattn_override", False):
            raise RuntimeError("SolAttn: another optimized_attention_override is already active on this MODEL.")
        transformer_options["optimized_attention_override"] = solattn_attention_override
        transformer_options[SPARSE_KEY] = True
        transformer_options[TAU_KEY] = float(tau)
        transformer_options[START_PERCENT_KEY] = float(start_percent)
        transformer_options[END_PERCENT_KEY] = float(end_percent)
        transformer_options[MIN_TOKENS_KEY] = int(min_tokens)
        # dense_blocks 按模型总层数解析（负数从末尾数）
        diffusion = model.model.diffusion_model
        n_blocks = len(getattr(diffusion, "blocks", None) or ())
        transformer_options[DENSE_BLOCKS_KEY] = parse_blocks(dense_blocks, n_blocks)
        transformer_options[PREFIX_TOKENS_KEY] = int(h3_prefix_tokens)
        transformer_options[TOPK_KEY] = int(topk_blocks)
        _install_block_index(diffusion)
        LOGGER.info("[SolAttn][V1.1.1] active: fp16_safe=%s tau=%.2f window=[%.2f,%.2f] "
                    "min_tokens=%d dense_blocks=%s prefix=%d topk=%d",
                    bool(fp16_safe), float(tau), float(start_percent), float(end_percent),
                    int(min_tokens), dense_blocks or "(none)", int(h3_prefix_tokens),
                    int(topk_blocks))
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "SolAttnV100": SolAttnV100,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SolAttnV100": "Sol-Attn (V100)",
}
