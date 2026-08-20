# Notices

本目录（`native/`）的源码与第三方库来源声明：

## 直接来源：H3_V100（rwashy/H3-V100）

`csrc/flash_attn/` 的 kernel 源码直接复制自
**https://github.com/rwashy/H3-V100** 的 `native/csrc/`（本地开发克隆，路径从略）。

该上游仓库**整体许可是 GPL-3.0-only**（因其 H3 mixed-precision 部分继承自
Icbears/minimax-h3-v100-patch）。但其中 **FlashAttention CUDA 组件为 BSD 3-Clause**
（上游 NOTICE.md 与 `licenses/FLASH_ATTENTION_BSD-3-CLAUSE.txt` 声明）。

**本仓库仅复制了该 BSD-3-Clause 组件**（sparse kernel 的 `.h/.cuh/.cu` 与 API 封装），
**未包含任何 GPL-3.0 部分**（如 `h3_mixed_precision.py` 等混合精度代码）。
本插件整体按 MIT 许可分发（见仓库根 LICENSE）。

## FlashAttention sparse kernel（`csrc/flash_attn/`）

`csrc/flash_attn/src/` 下的 CUDA kernel 源码来自 **flash-attention**（Tri Dao）的
sparse attention kernel（keep-or-drop），经 V100（sm_70）移植链分发：

- 上游官方仓库：https://github.com/Dao-AILab/flash-attention
  （flash-attention 采用 BSD 3-Clause 许可证）
- sm70 移植（V100 FlashAttention）：https://github.com/Icbears/flash-attention-v100
  （BSD 3-Clause）
- 本仓库仅做 sparse-only 精简（`flash_api.cpp` / `flash_api_torch_lib.cpp` /
  `setup.py`），kernel 源码保留原样与原始版权声明
  （`// Copyright (c) 2024, Tri Dao.`），便于与上游核对。

## CUTLASS（`third_party/cutlass/`）

`third_party/cutlass/` 为 **NVIDIA CUTLASS**（编译 kernel 所需头文件）：

- 官方仓库：https://github.com/NVIDIA/cutlass
- 许可证：Apache License 2.0，完整文本见 https://www.apache.org/licenses/LICENSE-2.0
