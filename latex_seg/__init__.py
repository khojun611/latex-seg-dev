# Copyright (c) Facebook, Inc. and its affiliates.
from .config import add_latexseg_config
from .semantic_seg import LaTeXSegHead
from .implicit_glyph_refinement import ImplicitGlyphRefinement
from .text_fpn import TextSemanticSegmentor, TextSemSegFPNHead
from .color_augmentation import ColorAugSSDTransform
from . import convnext_backbone as _  # backbone registration
from . import dinov3_backbone as _  # backbone registration
from . import dinov3_convnext_backbone as _  # backbone registration
