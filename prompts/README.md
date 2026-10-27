# GPT prompts

One `<image-stem>.txt` per TextSeg image (train 2646, test 1038), three numbered lines:
location, style and recognized text (`train_net.read_prompt_triplet`).

The dataset registry in `train_net.py` reads them from `$DETECTRON2_DATASETS/TextSeg/{train_gpt,test_gpt}`:

```
ln -s $PWD/prompts/TextSeg/train_gpt $DETECTRON2_DATASETS/TextSeg/train_gpt
ln -s $PWD/prompts/TextSeg/test_gpt  $DETECTRON2_DATASETS/TextSeg/test_gpt
```

Images and masks are not included; get TextSeg from its authors.
