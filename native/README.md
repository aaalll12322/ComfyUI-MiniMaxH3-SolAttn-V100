# native — SolAttn (V100) 稀疏 kernel 源码编译

本目录包含 keep-or-drop sparse kernel 的**完整源码**（flash-attn 官方 sparse kernel，Tri Dao 版权，sm70 适配），
用于编译 `comfy_v100_solattn_cuda*.pyd`。**仅含 `varlen_fwd_sparse` 一个 op**（无 dense/backward op）。

## 为什么源码编译

开源透明、可审计、平台自适配（跨平台/跨 Python 版本自行编译）。仓库**不提交 pyd 二进制**（`.gitignore`
排除），安装后本地编译一次即可。没有编译环境的用户可等 Release 预编译附件。

## 编译要求

- NVIDIA CUDA Toolkit（本机验证 CUDA 12.8）+ MSVC（Visual Studio Build Tools，C++17）
- PyTorch（带 CUDA，本机验证 2.8.0+cu128，Python 3.12）
- Windows SDK（rc.exe，`WindowsSdkBuildExtension` 自动定位 `C:\Program Files (x86)\Windows Kits\10\bin`）

## 编译命令

```bash
cd native
MAX_JOBS=4 python setup.py build_ext --inplace
```

## 产物文件名（重要）

输出文件名的后缀随**编译用的 Python 版本**变化：

| 编译用 Python | 产物文件名 |
|---|---|
| 3.12 | `comfy_v100_solattn_cuda.cp312-win_amd64.pyd` |
| 3.11 | `comfy_v100_solattn_cuda.cp311-win_amd64.pyd` |
| 3.13 | `comfy_v100_solattn_cuda.cp313-win_amd64.pyd` |

**插件启动时自动匹配 `comfy_v100_solattn_cuda` 前缀的任意 pyd**（`nodes.py` 的
`_load_extension` 用 glob 查找），所以**文件名不要求特定 Python 版本后缀**——编译出的
`cp311`/`cp313` 等版本直接复制到插件根目录即可被识别，无需改名。

> Windows 多核并行：`MAX_JOBS=4`（默认串行单核很慢）。

## 源码结构

```
native/
├─ setup.py                          # CUDAExtension（name=comfy_v100_solattn_cuda，仅 sparse 源）
├─ third_party/cutlass/              # CUTLASS（28MB，编译必需）
└─ csrc/
   ├─ common/                        # registration.h / pytorch_shim.h（op 注册辅助）
   └─ flash_attn/
      ├─ flash_api.cpp               # mha_varlen_fwd_sparse 实现（从 flash-attn 官方 API 精简）
      ├─ flash_api_torch_lib.cpp     # TORCH_LIBRARY 注册（仅 varlen_fwd_sparse）
      └─ src/                        # flash-attn 官方 kernel 头 + flash_fwd_sparse_hdim128_sm70.cu
```

## 版权与致谢

- kernel 源码直接来源：https://github.com/rwashy/H3-V100（该仓库整体 GPL-3.0-only，
  其中 FlashAttention CUDA 组件为 BSD 3-Clause；本仓库仅复制该 BSD 组件，无 GPL 部分。
  详见 [NOTICE.md](NOTICE.md)）。
- 上游：flash-attention（Tri Dao）官方 sparse kernel，
  https://github.com/Dao-AILab/flash-attention（BSD 3-Clause）；
  sm70 移植链经 https://github.com/Icbears/flash-attention-v100。
- CUTLASS：https://github.com/NVIDIA/cutlass（BSD 3-Clause，自 2025 年起；此前为 Apache-2.0），见 `third_party/cutlass/` 与根目录 `licenses/CUTLASS-BSD-3-CLAUSE.txt`。
- 稀疏算法/路由：Sol-Attn（arXiv 2607.24027，NVlabs/Sana sol-engine）。
