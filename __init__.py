# -*- coding: utf-8 -*-
"""Sol-Attn (V100) — MiniMax H3 在 V100 上的注意力稀疏加速（ComfyUI 自定义节点）。"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "1.0.0"
VERSION_TAG = "[SolAttn-V100][V1.0]"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__", "VERSION_TAG"]
