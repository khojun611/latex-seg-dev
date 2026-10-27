#!/bin/bash
# LaTeX-Seg on TextSeg: ResNet-101-FPN (ImageNet pretrained) +
# LLM-generated location/style/recognition text prompts.
# Datasets are read from $DETECTRON2_DATASETS (default: ./datasets); see train_net.py.
set -e

cd "$(dirname "$0")"

python train_net.py \
  --config-file configs/latexseg_R_101_FPN_textseg.yaml \
  --num-gpus 1 "$@"
