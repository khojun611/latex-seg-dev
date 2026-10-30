# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_latexseg_config(cfg):
    """
    Add config for LaTeX-Seg.
    """
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Color augmentatition from SSD paper for semantic segmentation model during training.
    cfg.INPUT.COLOR_AUG_SSD = False

    # Training-prompt mixing across LLMs: sibling dirs of the train prompt dir (e.g. train_gpt, train_gemini, ...).
    # Empty = only the dataset's own prompt dir. A source whose file is malformed (missing 1./2./3. lines, coordinate
    # or repetition garbage) is skipped for that image; the dataset's own prompt dir is the final fallback.
    # PER_LINE: draw location / style / recognition lines independently (4 sources -> 64 combinations per image).
    cfg.INPUT.PROMPT_MIX = CN()
    cfg.INPUT.PROMPT_MIX.SOURCES = ()
    cfg.INPUT.PROMPT_MIX.WEIGHTS = ()  # same length as SOURCES; empty = uniform
    cfg.INPUT.PROMPT_MIX.PER_LINE = False

    # Optimizer: "SGD" (detectron2 default) or "ADAMW".
    cfg.SOLVER.OPTIMIZER = "SGD"

    # Coarse head that LaTeXSegHead wraps.
    cfg.MODEL.SEM_SEG_HEAD.COARSE_HEAD_NAME = "TextSemSegFPNHead"

    # Hi-SAM-style text injection into the last NUM_BLOCKS blocks of the stride-16 backbone stage.
    cfg.MODEL.TEXT_INJECTION = CN()
    cfg.MODEL.TEXT_INJECTION.LOCATION = "backbone"  # "backbone" (stride-16 blocks) or "fpn" (p5/p4/p3, paper Eq. 4)
    cfg.MODEL.TEXT_INJECTION.NUM_BLOCKS = 3   # "backbone" only: semantic -> style -> location
    cfg.MODEL.TEXT_INJECTION.NUM_HEADS = 8
    cfg.MODEL.TEXT_INJECTION.GATE_INIT = 0.05
    # "backbone" only: explicit block indices (l, s, m) for location / style / semantic (Hi-SAM --pv_blocks);
    # empty = the last NUM_BLOCKS blocks (semantic -> style -> location).
    cfg.MODEL.TEXT_INJECTION.BLOCKS = ()

    # Text encoder: "clip" (OpenAI CLIP ViT-B/32 text, 512-d) or "dinotxt" (frozen dino.txt text tower aligned with
    # DINOv3 ViT-L/16; latex_seg/dinotxt_text.py). DINOTXT_PART: "patch" (patch-aligned half, 1024-d), "cls", "all" (2048).
    cfg.MODEL.TEXT_ENCODER = CN()
    cfg.MODEL.TEXT_ENCODER.NAME = "clip"
    cfg.MODEL.TEXT_ENCODER.DINOTXT_WEIGHTS = ""
    cfg.MODEL.TEXT_ENCODER.DINOTXT_PART = "patch"

    # Hi-SAM role heads on the injection blocks (latex_seg/role_heads.py; needs TEXT_INJECTION.BLOCKS):
    # location block -> text region, style block -> [glyph SDF, word SDF], semantic block -> skeleton.
    cfg.MODEL.ROLE_HEADS = CN()
    cfg.MODEL.ROLE_HEADS.ENABLED = False
    cfg.MODEL.ROLE_HEADS.REGION_WEIGHT = 1.0
    cfg.MODEL.ROLE_HEADS.SDF_WEIGHT = 0.5
    cfg.MODEL.ROLE_HEADS.SKEL_WEIGHT = 0.5

    # Implicit glyph refinement (IGR).
    cfg.MODEL.IGR = CN()
    cfg.MODEL.IGR.ENABLED = True  # False: coarse head only (logits upsampled x4 at inference, no loss_igr)
    cfg.MODEL.IGR.NAME = "ImplicitGlyphRefinement"
    cfg.MODEL.IGR.NUM_CLASSES = 2
    # Input feature map(s) of IGR.
    cfg.MODEL.IGR.IN_FEATURES = ("p2",)
    # Training points: N points sampled uniformly from the band |GDF| < GDF_BAND (paper Eq. 7).
    cfg.MODEL.IGR.GDF_BAND = 0.005
    cfg.MODEL.IGR.TRAIN_NUM_POINTS = 8192
    # Inference: number of x2 upsampling steps and max points refined per step.
    cfg.MODEL.IGR.SUBDIVISION_STEPS = 2
    cfg.MODEL.IGR.SUBDIVISION_NUM_POINTS = 20000

    # Glyph distance field (GDF) predictor (SIREN) on p2; L1 to the GT GDF from the mask.
    cfg.MODEL.GDF = CN()
    cfg.MODEL.GDF.LOSS_WEIGHT = 1.0
    cfg.MODEL.GDF.HIDDEN_DIM = 256
    cfg.MODEL.GDF.NUM_LAYERS = 3  # number of sine layers
    cfg.MODEL.GDF.W0 = 30.0       # SIREN omega_0

    # ConvNeXt V2 backbone (MODEL.BACKBONE.NAME = "build_convnextv2_fpn_backbone").
    cfg.MODEL.CONVNEXT = CN()
    cfg.MODEL.CONVNEXT.WEIGHTS = "facebook/convnextv2-large-22k-384"  # HF id or local dir
    cfg.MODEL.CONVNEXT.DROP_PATH_RATE = 0.0
    cfg.MODEL.CONVNEXT.FREEZE = False
    cfg.MODEL.CONVNEXT.GRAD_CHECKPOINT = False  # trade speed for memory
    cfg.MODEL.CONVNEXT.ADAPTERS = False  # Hi-SAM Space/MLP adapters in every block (use with FREEZE)

    # DINOv3 ViT backbone (MODEL.BACKBONE.NAME = "build_dinov3_vit_sfp_backbone").
    cfg.MODEL.DINOV3 = CN()
    cfg.MODEL.DINOV3.ARCH = "vitb16"
    cfg.MODEL.DINOV3.WEIGHTS = ""       # official checkpoint, e.g. dinov3_vitb16_pretrain_lvd1689m-*.pth
    cfg.MODEL.DINOV3.FREEZE = True      # Hi-SAM: train only the adapters
    cfg.MODEL.DINOV3.ADAPTERS = True
    cfg.MODEL.DINOV3.GRAD_CHECKPOINT = False
    # Space_Adapter residual outside the frozen LayerScale ls1 (zero-init); see dinov3_backbone.py.
    cfg.MODEL.DINOV3.SPACE_ADAPTER_OUTSIDE_LS = False
    # Frozen DINOv3 ConvNeXt stages 1-2 added to SFP p2 / p3 through zero-init projections (stride-4/8 detail).
    cfg.MODEL.DINOV3.CONV_STEM = CN()
    cfg.MODEL.DINOV3.CONV_STEM.ENABLED = False
    cfg.MODEL.DINOV3.CONV_STEM.SIZE = "large"
    cfg.MODEL.DINOV3.CONV_STEM.WEIGHTS = ""  # official .pth, .safetensors or HF model directory
    cfg.MODEL.DINOV3.CONV_STEM.LEVELS = ("p2", "p3")

    # RGB skip (latex_seg/text_fpn.py RGBRefiner): full-resolution residual on the x4-upsampled coarse logits from
    # the input image (conv3x3 x NUM_CONVS, last conv zero-init -> starts as plain bilinear upsampling). IGR off only.
    cfg.MODEL.RGB_SKIP = CN()
    cfg.MODEL.RGB_SKIP.ENABLED = False
    cfg.MODEL.RGB_SKIP.DIM = 32
    cfg.MODEL.RGB_SKIP.NUM_CONVS = 2

    # DINOv3 ConvNeXt backbone (MODEL.BACKBONE.NAME = "build_dinov3_convnext_fpn_backbone").
    cfg.MODEL.DINOV3_CONVNEXT = CN()
    cfg.MODEL.DINOV3_CONVNEXT.SIZE = "large"
    cfg.MODEL.DINOV3_CONVNEXT.WEIGHTS = ""  # official .pth, .safetensors or HF model directory
    cfg.MODEL.DINOV3_CONVNEXT.DROP_PATH_RATE = 0.0
    cfg.MODEL.DINOV3_CONVNEXT.FREEZE = True     # Hi-SAM: train only the adapters
    cfg.MODEL.DINOV3_CONVNEXT.ADAPTERS = True
    cfg.MODEL.DINOV3_CONVNEXT.GRAD_CHECKPOINT = False

    # Hi-SAM text-consistency loss (--cons_mode safe --cons_help): correct / no-text / wrong text.
    cfg.TEXT_CONSISTENCY = CN()
    cfg.TEXT_CONSISTENCY.ENABLED = True
    cfg.TEXT_CONSISTENCY.NOTEXT_SEG_WEIGHT = 1.0  # seg loss of the no-text forward
    cfg.TEXT_CONSISTENCY.SAFE_WEIGHT = 0.1        # relu(Q(wrong) - sg[Q(no-text)])
    cfg.TEXT_CONSISTENCY.HELP_WEIGHT = 0.5        # relu(Q(correct) - sg[Q(no-text)] + margin), text path only
    cfg.TEXT_CONSISTENCY.HELP_MARGIN = 0.2
    cfg.TEXT_CONSISTENCY.HELP_REL_MARGIN = 0.0    # > 0: relu(Q(correct) - (1-rel) sg[Q(no-text)]) instead (scale-free)
    cfg.TEXT_CONSISTENCY.CONTENT_WEIGHT = 0.0     # relu(Q(correct) - (1-rho) sg[Q(wrong)]) per image, text path only
    cfg.TEXT_CONSISTENCY.CONTENT_RHO = 0.1
    cfg.TEXT_CONSISTENCY.BUFFER = 64              # FIFO of past texts used as the wrong text
    cfg.TEXT_CONSISTENCY.NEG_BATCH = 2            # samples used for the extra forwards (memory limit)

    # Hi-SAM propose-and-verify structure heads on the coarse head's stride-4 feature (latex_seg/pv_heads.py).
    cfg.MODEL.PV = CN()
    cfg.MODEL.PV.ENABLED = False
    cfg.MODEL.PV.ANNO_DIR = ""            # TextSeg <stem>_anno.json (word quads + text); needed for word SDF / verify
    cfg.MODEL.PV.PROPOSAL_WEIGHT = 1.0    # text-region map, bce + dice
    cfg.MODEL.PV.WORD_SDF_WEIGHT = 0.5    # word signed distance, L1
    cfg.MODEL.PV.SKEL_WEIGHT = 0.5        # skeleton heat, bce + dice
    cfg.MODEL.PV.VERIFY_WEIGHT = 0.5      # GT word region x prompt word, balanced BCE
    cfg.MODEL.PV.MAX_WORDS = 16           # prompt words per image
