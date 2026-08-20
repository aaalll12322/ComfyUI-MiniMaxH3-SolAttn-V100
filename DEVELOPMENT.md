# DEVELOPMENT — ComfyUI-MiniMaxH3-SolAttn-V100 开发与设计文档

> English: [DEVELOPMENT_EN.md](DEVELOPMENT_EN.md) · 使用文档: [README.md](README.md) / [README_EN.md](README_EN.md)
>
> **AI 辅助开发声明**：本项目由作者借助 AI 助手（DeepSeek-V4-Flash）完成开发，作者本人无计算机专业背景。本文档记录设计决策、验证数据与踩坑，供后续迭代与维护参考。

## 1. 目标

在 **V100（sm_70，单卡 16GB，无 bf16/fp8 硬件）** 上加速 MiniMax H3 视频生成的 attention 计算（480p 时占每步 ~79% 时间）。基线：纯 FP16Safe 71-74s/步（480p/10s）。

设计原则（用户明确要求）：
- **复用现成、验证过的实现**，不重造轮子（Sol-Attn 官方算法 + flash-attn 官方 sparse kernel）；
- **单节点**：一个 `Sol-Attn (V100)` 节点 = FP16Safe（fp16 安全）+ Sol-Attn 稀疏，工作流不再需要串联多个节点；
- 一切性能/质量结论以**真机实测**为准（GPU util/power/显存监控 + 用户真机视频判定）。

## 2. 架构

```
Sol-Attn (V100) 节点（单节点 patch）
├─ fp16_safe=True → 内嵌 FP16Safe（_load_fp16safe_nodes 扫 custom_nodes 加载其 nodes.py）
│    prescale x/16 → qkv_proj → attention → out_proj 后 fp32 ×16 还原
│    deferred isfinite 熔断：整 forward fp32 重跑兜底
├─ optimized_attention_override（分发入口）
│    ├─ 满足稀疏条件 → routing.py（kc/vc + diag 阈值 + 邻域±1 + sink）
│    │    → CSR 掩码 → varlen_fwd_sparse（keep-or-drop kernel，源码 native/ 或预编译 pyd）
│    └─ 不满足（mask/非 fp16/短序列/窗口外/dense_blocks）→ 原版 SDPA 兜底
└─ 稀疏防线：采样窗口（start_percent/end_percent）+ dense_blocks（保留 dense 层）
              + h3_prefix_tokens（KV sink 保底）
```

### 组件来源（全部复用现成实现）

| 组件 | 来源 | 说明 |
|---|---|---|
| `routing.py` | Sol-Attn 官方算法（[arXiv 2607.24027](https://arxiv.org/abs/2607.24027)，NVlabs sol-engine preprocess.py 复刻） | BLOCK=64、kc 块均值、vc 块和、diag 阈值、列均值路由、CSR 生成 |
| `comfy_v100_solattn_cuda.pyd` | 本仓库 native 源码编译（flash-attn 官方 sparse kernel 的 sm70 适配，sparse-only） | 仅 `varlen_fwd_sparse` 一个 op；源码在 `native/`，Release 附预编译 |
| FP16Safe | ComfyUI-MiniMaxH3-FP16Safe v6.8.0（逻辑内嵌 `fp16safe.py`） | 自包含：`_load_fp16safe_nodes` 直接 import 内置模块，无需外部插件 |

## 3. 关键设计决策

1. **稀疏 = keep-or-drop（per-head 独立）**：官方 Sol-Attn 的路由 + sparse kernel 精确计算选中块、跳过未选中块。实验证明在 H3 上块均值 zeroth-order 近似（kijai 融合两级）反而是负优化（4.3e-2 vs keep-or-drop 5.8e-3 的旧对比；最终以真机肉眼为准）。
2. **kernel 用 PAI 分支版而非官方版**：上游 fork 自带 sparse kernel 的 P 布局转换（手写 shfl 版）在 sm70 输出全零；PAI/Alibaba 分支用的 `convert_layout_C_to_A_v2`（复用 dense kernel 模板）正确。结论：**复用 = 完整实现 + 真机验证，不是只看代码**。
3. **dense 兜底 = 原版 SDPA**：大序列 + 低显存下，额外 kernel 的显存峰值换页会吃掉计算收益 → 保持最小内存足迹。用户实测 43s/步（比带额外 kernel 的组合快）。教训：**kernel 快 ≠ 端到端快，测速必须真机端到端**。
4. **路由向量化**：csr_from_sel 最初有 `for r in uniq` Python 循环（1562ms @ S=6154）；改为高级索引 + cumsum 槽位后 **10ms @ S=29650**。GPU 任务必须向量化（用户铁律）。
5. **`_sparse_attn` 不限制 S%64**：kernel（is_even_MN=false 路径）支持任意序列长度的边界处理（S=6154 实测正确）。
6. **采样窗口用 percent 语义**（kijai 风格）：`start_percent/end_percent` 由 transformer_options 的 `step/total_steps` 换算，缺省时回退 sigma>14 判断（turbo 4-step 第一步 sigma≈14.64 → 自然 dense）。

## 4. 验证数据（本机 V100-SXM2-16GB, torch 2.8.0+cu128, ComfyUI Python 3.12）

### 4.1 kernel 数学正确性（vs PyTorch keep-or-drop 模拟）

| 测试 | rel-L2 |
|---|---|
| 全选掩码（=dense 语义） | 3.0e-4 |
| 单块 S=64 | 5.9e-6 |
| 每行不同掩码 S=256/1024/2048/4096（随机） | 2.1-2.7e-4 |
| S=6154 真实激活（含非 64 倍数边界） | 2.1e-4 |

### 4.2 速度（真实激活 / 480p 规模）

| 场景 | 耗时 | 倍率 |
|---|---|---|
| SDPA（S=6154） | 30.3ms | 1× |
| sparse kernel（S=6154） | 12.3ms | 2.46× |
| SDPA（S=29650） | 814ms | 1× |
| sparse kernel（S=29650） | 179ms | 4.55× |
| 路由+kernel（S=29650） | 294ms | 2.77× |

### 4.3 端到端（用户真实 ComfyUI，480p/10s）

| 方案 | s/步 | vs 基线 |
|---|---|---|
| 纯 FP16Safe | 71-74 | 1× |
| **Sol-Attn v1.0（tau=1.0）** | **43** | **~1.7×，画质肉眼无损** |

### 4.4 路由优化

| 版本 | 路由耗时 @S=29650 | 说明 |
|---|---|---|
| v1.0（nonzero 循环） | 1562ms | 逐行 Python 循环 |
| v1.0（scatter+masked_fill 初版） | 115ms | **有 bug**：`masked_fill_(~sel_perm)` 清的是列位置、scatter 写的是槽位位置 → 未选中列污染槽 0 → kernel vs sim rel 2.67 |
| **v1.0（高级索引）** | **10ms** | 只写选中位置；正确性恢复 rel 2.85e-4 |

另：kernel 直接吃非连续 q/k/v（按 stride 访问，实测更快）→ 省 3 次 contiguous 拷贝（~7ms）。

### 4.5 稀疏质量（诚实记录）

per-head keep-or-drop，τ=1.0 时密度 26.4%（S=6154 真实激活），vs dense 的 rel-L2 = **0.223**。早期记录的"5.8e-3"是 head 并集计算的假象（路由 per-head 选块、计算取并集 → 实际密度远高于报告值），度量不一致已作废。**尽管 rel 0.22，真实视频（turbo 4step）肉眼无损**——最终质量以真机判定为准（用户要求，替代纯 L∞/rel 指标）。

## 5. 踩坑记录

1. **diff 行尾符污染**：CRLF/LF 混用会让 diff 把整个文件标为不同 → 必须 `diff --strip-trailing-cr`。
2. **nvcc 模板错误行号偏移**：报错行号与源文件差 2 是 nvcc 对模板实例化错误的报告偏移，不要据此怀疑文件被旧缓存编译。
3. **作用域报错**：`rows_this_block`/`warp_row_base` 定义在嵌套块内、引用在外 → 内联为表达式解决。
4. **沙箱回收站**：`setup.py build_ext --inplace` 最后复制 pyd 时 safe-delete 被沙箱拦截（recycle-bin 不可用）→ 编译成功后手动 `cp` 产物（build/lib.win-amd64-cpython-312/*.pyd）。
5. **ninja 异常退出（0x40000004）**：`_bt`/`build` 状态损坏 → 完全清理两目录重编。
6. **编译并行**：默认串行 CPU 跑不满 → `MAX_JOBS=4` 并行（4m48s vs 串行 4m+ 单核）。
7. **调试 printf**：kernel 内的 SPARSE_* printf 是调试残留，上线前必须删除（性能 + 刷屏）。
8. **scatter+masked_fill 陷阱**：`off.scatter_(1, rank, col)` 写"槽位"，`masked_fill_(~sel)` 清"列位"——两者坐标系不一致会互相污染。用高级索引 `off[rows, pos] = col[cols]` 只写选中位置最稳。
9. **显存峰值换页吃收益（大序列 + 低显存）**：kernel 快 ≠ 端到端快；额外 kernel 的显存峰值导致模型换页，可吃掉全部计算收益。测速必须端到端真机，不能只看 kernel 单测。
10. **组件移除不彻底**：重构/改名时，除删除相关模块文件外，还要清理 nodes.py 中的 import、transformer_options 键、参数与日志引用——逐个 grep 确认无残留。

## 6. 已知限制与后续方向

- **路由开销**：Python 层 ~10ms @ S=29650，理论 GEMM 级 ~1ms。优化方向：torch.compile / kernel 内联路由（kijai 融合两级结构）。
- **稀疏质量边界**：H3 注意力分散（top-16/97 块仅 66% mass），要保 95% mass 需 ~48% 密度。当前 26% 密度肉眼无损（turbo 场景），更敏感场景需调参（tau/end_percent/dense_blocks）或加 zeroth-order 修正项。
- **tau_profile（per-block tau）**：kijai 支持按层配置 tau（敏感层低 τ、迟钝层高 τ），当前为全局 tau，后续可加。
- **平台**：仅 V100/sm_70 + Windows + cp312 验证；pyd 需在目标平台重新编译。
- **候选路线**：①自研 keep-or-drop CUTLASS kernel（绕开 PAI 实验代码）；②对照 NVlabs/Sana sol-engine `models/minimax_h3/A100/adapter.py`（官方 H3 适配 3.95×-4.52×）；③等官方参考实现。
