# [IEEE TIP] Language-Guided Scene Text Segmentation via Hierarchical Text Context Reasoning and Implicit Glyph Refinement

Ho Jun Kim, Hak Gu Kim

Official implementation of **LaTeX-Seg** (**La**nguage-driven scene **TeX**t **Seg**mentation), built on [detectron2](https://github.com/facebookresearch/detectron2).

<p align="center">
  <img src="assets/teaser.png" width="60%">
</p>

Scene text is hard to segment from visual cues alone: stylized glyphs, logo-like distractors and thin boundaries
confuse existing models. LaTeX-Seg asks a multimodal LLM *where* the text is, *how* it looks and *what* it says,
and uses these answers, together with an explicit glyph geometry representation, to guide segmentation.

## Method

<p align="center">
  <img src="assets/overview.png" width="100%">
</p>

- **Hierarchical Text Context Reasoning (HTCR)** prompts an MLLM for three levels of context: **location**,
  **style** and **semantics** (the recognized text). Each answer is encoded token-wise by a CLIP text encoder.
- **Glyph Distance Field (GDF)**: a SIREN predictor gives every pixel its distance to the nearest text boundary,
  a geometry-aware representation of the glyph shape.
- **Progressive Text Context and Geometry Fusion**: the three text contexts and the GDF feature are injected into
  the multi-scale visual features by cross-attention, coarse to fine.
- **Implicit Glyph Refinement (IGR)** refines the coarse segmentation with an implicit, coordinate-based decoder
  to recover thin strokes and sharp boundaries.

<p align="center">
  <img src="assets/htcr.png" width="85%">
</p>
<p align="center"><em>HTCR: location suppresses distant logo patterns, style removes text-like background,
semantics recovers complete text regions.</em></p>

## Results

Foreground IoU / F-score (from the paper):

| Method | ICDAR13 FgIoU | ICDAR13 F | Total-Text FgIoU | Total-Text F | TextSeg FgIoU | TextSeg F |
|---|---|---|---|---|---|---|
| Hi-SAM | – | – | 84.86 | 89.68 | 88.96 | 93.87 |
| ConText | 75.90 | 87.30 | 85.19 | 91.90 | 89.74 | 94.60 |
| **LaTeX-Seg** | **76.32** | 86.51 | **85.66** | **92.28** | **90.23** | **94.86** |

Boundary IoU / boundary F-score:

| Method | ICDAR13 BIoU | ICDAR13 BF | Total-Text BIoU | Total-Text BF | TextSeg BIoU | TextSeg BF |
|---|---|---|---|---|---|---|
| Hi-SAM | 68.74 | 75.33 | 81.33 | 90.00 | 86.81 | 93.11 |
| ConText | 68.90 | 75.41 | 81.20 | 89.93 | 86.26 | 92.62 |
| **LaTeX-Seg** | **69.89** | **75.89** | **81.72** | **90.24** | **87.80** | **93.63** |

<p align="center">
  <img src="assets/qualitative.png" width="100%">
</p>

## Code

| File | Content |
|---|---|
| `latex_seg/text_fpn.py` | Text-conditioned segmentor: CLIP token features of the three prompts, gated cross-attention text injection (into the backbone, or progressively on the FPN levels), FPN head with the SIREN GDF predictor |
| `latex_seg/implicit_glyph_refinement.py` | Implicit glyph refinement (IGR) |
| `latex_seg/semantic_seg.py` | Coarse segmentation + IGR refinement (`LaTeXSegHead`) |
| `latex_seg/convnext_backbone.py` | ConvNeXt V2 backbone (Hugging Face weights), optional frozen backbone + adapters |
| `latex_seg/dinov3_convnext_backbone.py` | DINOv3 ConvNeXt backbone (loads the official / Hugging Face checkpoints), frozen + adapters |
| `latex_seg/dinov3_backbone.py` | DINOv3 ViT backbone with a simple feature pyramid, frozen + adapters |
| `train_net.py` | Dataset registration, text-prompt loader, text-consistency loss, trainer |

### Backbones

| Config | Backbone | Training | Text injection |
|---|---|---|---|
| `latexseg_R_101_FPN_textseg.yaml` | ResNet-101 (ImageNet) | full fine-tuning | last three stride-16 blocks |
| `latexseg_ConvNeXtV2_L_FPN_textseg.yaml` | ConvNeXt V2-L (FCMAE, IN-22k) | full fine-tuning | last three stride-16 blocks |
| `latexseg_ConvNeXtV2_L_Adapter_FPNText_textseg.yaml` | ConvNeXt V2-L (FCMAE, IN-22k) | frozen + adapters | FPN p5 / p4 / p3 (location / style / semantics) |
| `latexseg_DINOv3_ConvNeXt_L_Adapter_FPNText_textseg.yaml` | DINOv3 ConvNeXt-L (LVD-1689M) | frozen + adapters | FPN p5 / p4 / p3 |
| `latexseg_DINOv3_ViTB16_SFP_textseg.yaml` | DINOv3 ViT-B/16 (LVD-1689M) | frozen + adapters | last three blocks |

In the adapter variants the pretrained backbone is frozen and every block gets two small bottleneck adapters
(after the depthwise convolution and in parallel with the MLP, as in Hi-SAM). Only the adapters, the FPN, the
text fusion modules and the segmentation head are trained.

## Requirements

detectron2, [CLIP](https://github.com/openai/CLIP), `peft`, `transformers`, `safetensors`, `scipy`, `natsort`, `opencv-python`

## Data

Datasets are read from `$DETECTRON2_DATASETS` (default: `./datasets`):

```
datasets/
  TextSeg/
    train/  test/                 images
    seg_train/  seg_test/         masks (text=255, background=0)
    train_gpt/  test_gpt/         text prompts, one <image-stem>.txt per image
```

Each prompt file has three lines: location, style, recognition.

## Pretrained weights

- ConvNeXt V2: `MODEL.CONVNEXT.WEIGHTS` takes a Hugging Face model id or a local directory
  (default `facebook/convnextv2-large-22k-384`).
- DINOv3: the weights are released by Meta under the DINOv3 license and require access approval
  ([facebook/dinov3-convnext-large-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m),
  [DINOv3 repository](https://github.com/facebookresearch/dinov3)). Download them yourself and set
  `MODEL.DINOV3_CONVNEXT.WEIGHTS` (a Hugging Face model directory, `.safetensors` or the official `.pth`)
  or `MODEL.DINOV3.WEIGHTS`. They are not distributed with this repository.

## Training

```bash
# ResNet-101
python train_net.py --config-file configs/latexseg_R_101_FPN_textseg.yaml --num-gpus 1

# ConvNeXt V2-L, frozen backbone + adapters, text fused on the FPN
python train_net.py --config-file configs/latexseg_ConvNeXtV2_L_Adapter_FPNText_textseg.yaml --num-gpus 1

# DINOv3 ConvNeXt-L, frozen backbone + adapters, text fused on the FPN
python train_net.py --config-file configs/latexseg_DINOv3_ConvNeXt_L_Adapter_FPNText_textseg.yaml --num-gpus 1 \
  MODEL.DINOV3_CONVNEXT.WEIGHTS /path/to/dinov3-convnext-large-pretrain-lvd1689m
```

The adapter configs use batch size 8 with gradient checkpointing (about 21 GB on one GPU) and global
gradient-norm clipping (`SOLVER.CLIP_GRADIENTS.CLIP_TYPE: full_model`).

Evaluation: add `--eval-only MODEL.WEIGHTS <checkpoint>`.

## License

Apache License 2.0 (see `LICENSE`). Parts of the code are adapted from [detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0).
Pretrained backbone weights are subject to their own licenses.
