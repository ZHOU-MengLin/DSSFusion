# DSSFusion

Official code for **DSSFusion**: a dual-stream infrared–visible image fusion framework
with mask-guided foreground/background decomposition.

This repository contains the **core training and inference code**.

## Repository structure

```
DSSFusion_release/
├── arguments.py            # training / dataset hyper-parameters
├── train.py                # alternating fusion + segmentation training
├── test.py                 # inference / fusion result generation
├── module/
│   ├── FusionNet.py        # FusionNet, Encoder, Merge, SegmentMI
│   ├── TransformerNet.py   # SDIB (background branch)
│   ├── CBAM_old.py         # DAFB (foreground branch)
│   ├── FreqFusion.py       # frequency-aware fusion block
│   ├── model_TII.py        # BiSeNet segmentation network
│   └── resnet.py           # ResNet18 backbone for BiSeNet
└── utils/
    ├── dataset.py          # Fusion_dataset / CityScapes
    ├── loss.py             # Fusionloss, int_loss, int_loss_with_mask
    ├── optimizer.py        # SGD + warmup / poly schedule
    ├── logger.py
    ├── transform.py
    └── utils.py            # YCrCb conversion, seed, io helpers
```

## Environment

- Python >= 3.8
- PyTorch >= 1.10 (CUDA recommended)
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

## Dataset

By default the code expects an MSRS-style layout under `./datasets/MSRS`:

```
datasets/MSRS/
├── Visible/train/*.png
├── Infrared/train/*.png
├── Label/train/*.png
├── Visible/*.png        # test images (point --vi_dir here)
└── Infrared/*.png       # test images (point --ir_dir here)
```

You can override the paths in `arguments.py` (training) or via command line
(inference). Training pairs are loaded from `Visible/train`, `Infrared/train`
and `Label/train`; test images are read directly from the directories given by
`--vi_dir` / `--ir_dir`.

## Training

The training pipeline alternates **fusion training → fused-image generation →
segmentation training** for 4 rounds:

```bash
python train.py
```

Checkpoints and intermediate fused images are written to `model/Fusion` and
`Dataset/Fusion`.

## Inference

```bash
python test.py \
  --F_model_path model/Fusion/fusion_model_3.pth \
  --S_model_path model/Fusion/seg_model_3.pth \
  --vi_dir ./datasets/MSRS/Visible \
  --ir_dir ./datasets/MSRS/Infrared \
  --save_dir ./results/fusion
```

## Loss

The masked intensity loss is defined in `utils/loss.py`:

```
L = (1 + γ) · L_bg + (1 - γ) · L_fg,   γ = 0.6
```

where `L_fg` / `L_bg` are the normalized mean absolute errors between the fused
image and `max(VI, IR)` over foreground / background regions.

## Citation

If you find this work useful, please cite the corresponding paper.
