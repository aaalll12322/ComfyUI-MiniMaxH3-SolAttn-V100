# ComfyUI-MiniMaxH3-SolAttn-V100

**Sol-Attn（arXiv 2607.24027）稀疏加速 · MiniMax H3 在 V100 上的单节点注意力加速插件（ComfyUI 自定义节点）· v1.0.0**

> English version: [README_EN.md](README_EN.md) · 开发/设计文档: [DEVELOPMENT.md](DEVELOPMENT.md) / [DEVELOPMENT_EN.md](DEVELOPMENT_EN.md)
>
> **AI 辅助开发声明**：本项目由作者借助 AI 助手（DeepSeek-V4-Flash）完成开发，作者本人无计算机专业背景，代码与文档均由 AI 辅助编写。项目按 MIT 协议以现状分享；使用中如有问题，欢迎提 Issue，我们会在能力范围内协助解决。

一个节点 = **内嵌 FP16Safe（fp16 安全）+ Sol-Attn 官方稀疏（keep-or-drop）**。在 V100（sm_70）上跑 MiniMax H3 视频生成，480p/10s 从 **71-74s/步 降到 43s/步（约 1.7×），画质肉眼无损**。

---

## 引用与使用的内容（Credit）

本项目是**现成、验证过的实现**的组合，核心加速全部来自：

| 组件 | 来源 | 用途 |
|---|---|---|
| **Sol-Attn 稀疏算法** | [arXiv 2607.24027](https://arxiv.org/abs/2607.24027)（NVlabs/Sana sol-engine，`techniques/sparse_backends/sol_attn/preprocess.py`） | kc/vc 块统计 + diag 阈值 + 列均值路由 + keep-or-drop（`routing.py` 按官方算法复刻，per-head 独立路由） |
| **keep-or-drop kernel** | 直接来源 [rwashy/H3-V100](https://github.com/rwashy/H3-V100)（整体 GPL-3.0-only，其中 FlashAttention CUDA 组件 BSD 3-Clause；仅复制 BSD 组件）；上游 [flash-attention](https://github.com/Dao-AILab/flash-attention)（Tri Dao）+ [Icbears/flash-attention-v100](https://github.com/Icbears/flash-attention-v100) | 选中块精确计算、未选中块跳过（`comfy_v100_solattn_cuda`：源码在 `native/`，或 Release 预编译 pyd） |
| **fp16 NaN 安全** | [ComfyUI-MiniMaxH3-FP16Safe](https://github.com/aaalll12322/ComfyUI-MiniMaxH3-FP16Safe) v6.8.0（逻辑内嵌为 `fp16safe.py`） | prescale /16 + 熔断 + fp32 重跑兜底（**自包含，无需单独安装 FP16Safe 插件**） |
| **参数风格** | [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) | tau / start_percent / end_percent / dense_blocks / sink 参数体系对齐 |

---

## 解决的问题

MiniMax H3 在 V100 上的两个核心瓶颈：

1. **attention 是绝对大头**（480p 时占每步 ~79% 时间），而 PyTorch SDPA 在 V100 上仅 ~37T 吞吐；
2. **fp16 计算会 NaN**（H3 激活值真实可达 50 万，远超 fp16 上限），官方只支持 bf16/fp32，V100 无 bf16 硬件只能回落 fp32（慢 4×）。

Sol-Attn 稀疏的核心思路：attention 大多数 score 是噪声，**只对少量高价值 KV 块做精确计算**（keep-or-drop），其余直接跳过——省掉 O(n²) 的大头。

---

## 实测性能（用户真实 ComfyUI，480p/10s）

| 方案 | 每步耗时 | 画质 |
|---|---|---|
| 纯 FP16Safe（基线） | 71-74s | 正常 |
| 纯 SDPA + FP16Safe（dense 兜底） | ~70s | 正常 |
| **Sol-Attn（推荐，`tau=1.0`）** | **43s** | **肉眼无损** |

- vs 纯 FP16Safe：**~1.7×**
- 单次 attention（S=29650）：sparse kernel 比 SDPA 快 **4.55×**（路由+kernel 合计 2.77×）
- 路由已向量化：**10ms @ S=29650**；kernel 直接吃非连续输入省 3 次拷贝

---

## 安装

```bash
cd ComfyUI/custom_nodes
# 方式一：git clone（若已发布到 GitHub）
git clone https://github.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100.git
# 方式二：手动复制整个文件夹到 custom_nodes/（需要 kernel：见下方"kernel 获取"）
```

**kernel 获取**（`comfy_v100_solattn_cuda*.pyd`，二选一；插件启动时自动匹配 `comfy_v100_solattn_cuda` 前缀的 pyd，**文件名不要求特定 Python 版本后缀**）：
- **Release 预编译**：从 [GitHub Release](https://github.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100/releases) 下载（Windows + Python 3.12，零编译）
- **源码编译**：仓库自带完整源码（`native/`，含 CUTLASS），一条命令：
  ```bash
  cd native && python setup.py build_ext --inplace
  # 产物（comfy_v100_solattn_cuda.cpXXX-win_amd64.pyd，XXX 随编译用的
  # Python 版本变化）复制到插件根目录即可，插件自动识别
  ```

重启 ComfyUI。工作流中在 `sol_attn` 分类下找到 **"Sol-Attn (V100)"** 节点。

**依赖**：
- **ComfyUI**（含 `comfy/ldm/minimax`，PR #15224）
- **NVIDIA V100（sm_70）**——其他架构启动时报错
- **无额外 Python 包、无额外插件**：FP16Safe 逻辑内嵌（`fp16safe.py`），CUTLASS kernel 源码随仓库分发（`native/`），预编译 pyd 从 Release 获取

---

## 使用

```
H3 模型 ──> Sol-Attn (V100) ──> 采样器（KSampler 等）
```

单节点同时完成 fp16 安全 + 稀疏，**不需要**再串联 FP16Safe 节点。

### 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| `fp16_safe` | true | 内嵌 FP16Safe（prescale /16 + 熔断 + fp32 重跑兜底）。关闭需工作流中另有 fp16 安全措施 |
| `tau` | 1.0 | 稀疏路由阈值 β：越大越稀疏越快。1.0 ≈ 26% 密度（V100 fp16 实测画质肉眼无损），1.5 更低密度更快 |
| `start_percent` | 0.2 | 采样进度低于此比例走 dense（论文用 0.2；turbo 4-step 下第一步自然 dense） |
| `end_percent` | 1.0 | 采样进度高于此比例走 dense（1.0=不启用尾部 dense，与 43s/步 实测一致；调 0.9 保尾部质量） |
| `min_tokens` | 1024 | 序列短于此 token 数不走稀疏（直接 SDPA） |
| `dense_blocks` | "0-1" | 保留 dense 的 transformer 块，如 `"0-1"`=前两层，`"0-2,-1"`=前三层+最后一层（-1 从末尾数）；空=全部稀疏 |
| `h3_prefix_tokens` | 0 | H3 序列 text/cond/ref/audio 前缀 token 数（KV sink 保底，建议填 634+cond+ref+audio 实际值；turbo 场景可留 0） |
| `debug_nan` / `profile` | false | 透传 FP16Safe 的 NaN 检测 / 耗时统计 |

### 推荐配置

- **推荐（480p 最快）**：`tau=1.0, start_percent=0.2, end_percent=1.0, dense_blocks="0-1"` → 43s/步（画质肉眼无损）
- **更激进**：`tau=1.2~1.5`（密度更低更快，画质需自行确认）
- **保守**：`dense_blocks="0-2,-1"` 或 `end_percent=0.9`（更多层/尾部走 dense，质量更稳，稍慢）
- **小分辨率**（608 及以下）：attention 占比低，稀疏收益小，建议 `end_percent=0` 全 dense 或不用本插件

---

## 原理（摘要）

1. **路由（routing.py，官方 Sol-Attn 算法）**：kc 块均值/vc 块和 → diag 阈值（key 空间解析投影）→ 列均值路由（| 邻域 ±1）→ CSR 掩码。per-head 独立，26% 密度下 rel-L2 0.22，但**真实视频肉眼无损**。
2. **kernel**：keep-or-drop sparse kernel（选中块精确计算，未选中块跳过），fp16 + head_dim 128，sm70 CUTLASS。
3. **FP16Safe**：x/16 prescale → qkv → attention → out_proj 后 /16 还原（fp32），deferred isfinite 熔断，触发则整 forward fp32 重跑。

---

## 已知限制

- 仅验证 **V100（sm_70）** + Windows + Python 3.12（cp312 pyd）；其他平台需自行编译 native。
- 稀疏路由在 Python 层（~10ms @ S=29650），仍有优化空间；kernel 本身很快（4.55×）。
- 稀疏质量以真机肉眼/PSNR 为准；极敏感场景（如精细文字）建议调高 `dense_blocks` / `end_percent` 或 `tau` 调低。
- dense 兜底 = 原版 SDPA。

---

## 版本历史

- **v1.0.0（2026-08-20）正式版**：单节点 = 内嵌 FP16Safe（`fp16safe.py`，v6.8.0 逻辑，自包含）+ Sol-Attn 稀疏（keep-or-drop，sparse-only kernel）。480p/10s 实测 **43s/步，画质肉眼无损**（~1.7× vs 纯 FP16Safe 71-74s）。参数对齐 kijai 风格（tau / start_percent / end_percent / min_tokens / dense_blocks / h3_prefix_tokens）；dense 兜底 = 原版 SDPA；kernel 源码在 `native/` 可自行编译，Release 提供预编译 pyd。版本串 `[SolAttn-V100][V1.0]`。

---

## Citation

本项目使用的稀疏算法来自 Sol-Attn 论文。如果本项目对你有帮助，请同时引用：

```bibtex
@article{solattn,
  title={Sol-Attn: Training-free Sparse Attention for Accelerating Image and Video Generation},
  author={NVlabs / Sana sol-engine team},
  journal={arXiv preprint arXiv:2607.24027},
  year={2026}
}
```

- 论文：https://arxiv.org/abs/2607.24027
- 官方代码：<https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends/sol_attn>（Sol-Attn 实现位于 `sol-engine` 分支）
- 项目页：https://nvlabs.github.io/Sana/Sol-Attn/
