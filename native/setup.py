import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
FLASH = CSRC / "flash_attn"

sdk_candidates = sorted(
    Path(r"C:\Program Files (x86)\Windows Kits\10\bin").glob(r"*\x64\rc.exe")
)
WINDOWS_SDK_BIN = sdk_candidates[-1].parent if sdk_candidates else Path()


class WindowsSdkBuildExtension(BuildExtension):
    def build_extensions(self):
        if not self.compiler.initialized:
            self.compiler.initialize()
        self.compiler._paths = os.pathsep.join(
            [str(WINDOWS_SDK_BIN), self.compiler._paths]
        )
        super().build_extensions()


# SolAttn (V100) keep-or-drop sparse kernel only（flash-attn 官方 sparse kernel 的 sm70 适配）。
# 仅编译 varlen_fwd_sparse 一个 op；无 dense/backward op。
extension = CUDAExtension(
    name="comfy_v100_solattn_cuda",
    sources=[
        "csrc/flash_attn/flash_api.cpp",
        "csrc/flash_attn/flash_api_torch_lib.cpp",
        "csrc/flash_attn/src/flash_fwd_sparse_hdim128_sm70.cu",
    ],
    include_dirs=[
        str(FLASH),
        str(FLASH / "src"),
        str(CSRC / "common"),
        str(ROOT / "third_party" / "cutlass" / "include"),
    ],
    define_macros=[
        ("FLASH_NAMESPACE", "comfy_v100_solattn"),
        ("FLASHATTENTION_DISABLE_BACKWARD", None),
        ("FLASHATTENTION_DISABLE_DROPOUT", None),
        ("FLASHATTENTION_DISABLE_ALIBI", None),
        ("FLASHATTENTION_DISABLE_SOFTCAP", None),
        ("FLASHATTENTION_DISABLE_LOCAL", None),
        ("FLASHATTENTION_DISABLE_PYBIND", None),
        ("COMFY_V100_HDIM128_256_ONLY", None),
        ("CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL", "1"),
        ("NOMINMAX", None),
    ],
    extra_compile_args={
        "cxx": ["/O2", "/std:c++17", "/Zc:preprocessor"],
        "nvcc": [
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-gencode=arch=compute_70,code=sm_70",
            "-Xcompiler=/Zc:preprocessor",
            "-t", "8",                    # 单 .cu 内部并行编译（默认 1 线程；CPU 利用率低的主因）
            "-DFLASH_NAMESPACE=comfy_v100_solattn",
            "-DFLASHATTENTION_DISABLE_BACKWARD",
            "-DFLASHATTENTION_DISABLE_DROPOUT",
            "-DFLASHATTENTION_DISABLE_ALIBI",
            "-DFLASHATTENTION_DISABLE_SOFTCAP",
            "-DFLASHATTENTION_DISABLE_LOCAL",
            "-DFLASHATTENTION_DISABLE_PYBIND",
            "-DCOMFY_V100_HDIM128_256_ONLY",
            "-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1",
            "-DNOMINMAX",
        ],
    },
)

setup(
    name="comfy-v100-solattn",
    version="1.0.0",
    ext_modules=[extension],
    cmdclass={"build_ext": WindowsSdkBuildExtension.with_options(use_ninja=True)},
)
