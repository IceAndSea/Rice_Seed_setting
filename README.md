# BiSSR-Net

Code for the paper **"BiSSR-Net: A bimodal oriented object detection framework for rice seed-setting rate estimation"**.

## Overview

BiSSR-Net detects filled and unfilled rice grains with oriented bounding boxes (OBBs) from paired reflective and transmissive images. The repository also contains MMSSR-Net, which handles paired, reflective-only, or transmissive-only inputs.

- **ModalitySelector**: splits the six-channel input into two three-channel streams
- **Dual-stream backbone**: two independently parameterized YOLOv12n backbones with A2C2f modules
- **CMFT-DEPA**: Cross-Modal Feature Transport (CMFT) followed by DEPA (dual spatial enhancing pixel weight assignment, adapted from DEYOLO)
- **OBB detection head**: oriented bounding box prediction

## Input format

The model accepts **six-channel images** with the following channel order:

| Channels | Modality     | Illumination |
|----------|--------------|--------------|
| `[0:3]`  | Transmissive | Backlight    |
| `[3:6]`  | Reflective   | Top light    |

Training images are organized as:

- `images/`: reflective images
- `imagesT/`: transmissive images (same file names)

## Installation

- Python >= 3.8
- PyTorch

The repository includes the modified `ultralytics` package (YOLOv12 with the BiSSR-Net modules). Run all scripts from the repository root.

## Training

Set `path` in `ultralytics/cfg/datasets/obb-seed.yaml` to the dataset root (class 0 = unfilled, class 1 = filled). `train_bissr.py` initializes the model from the pretrained `yolo12n.pt` weights.

BiSSR-Net (Setting 7 in the paper). `train_bissr.py` uses the model configuration `ultralytics/cfg/models/fuse/DEYOLOseed12.yaml` (`dea-seed-fuse.yaml` in the same folder defines the same layers):

```bash
python train_bissr.py
```

MMSSR-Net. `train_mmssr.py` uses the same model configuration and `SCHEDULE = "full_warmup_120"`, the four-stage schedule reported in the paper (Supplementary Table S1: epochs 1-120 paired only, 121-180 90/5/5, 181-255 80/10/10, 256-300 95/2.5/2.5 for paired/reflective-only/transmissive-only inputs):

```bash
python train_mmssr.py
```

Both scripts use the settings of the paper: 1024 x 1024 inputs, batch size 32, seed 0, SGD (initial learning rate 0.01, final factor 0.01, momentum 0.937, weight decay 0.0005, 3 warm-up epochs), FP32 and up to 300 epochs (patience 40), with mosaic enabled throughout training.

During MMSSR-Net training, each source image independently receives an input mode (paired, reflective-only, or transmissive-only) before mosaic augmentation. A mosaic can therefore contain tiles with different input modes. The first source image supplies the mosaic's modality flag for branch selection and for counting single-modality samples when scaling the batch loss.

## Full-image inference

Reflective and transmissive images of a pair are stored in the same directory and named `<sample>-1.<ext>` (reflective) and `<sample>-2.<ext>` (transmissive).

```bash
python val_bissr.py --weights path/to/best.pt --img_dir path/to/images --conf 0.2
python val_mmssr.py --weights path/to/best.pt --img_dir path/to/images --mode dual   # or trans / reflect
```

Inference uses 1024 x 1024 slices with 300-pixel overlap. After full-image coordinates are restored, the Edge Confidence Penalty is applied (score x 0.6 within 32 pixels of an internal slice edge), followed by cross-class IoF containment filtering (IoF > 0.85 and area ratio < 0.60) and class-agnostic global rotated NMS (IoU 0.5). Boxes with confidence >= 0.20 (`--conf`) are counted for the seed-setting rate. `--slice-conf` optionally sets a lower confidence floor for collecting slice predictions before merging.

## Loss weights

- `box`: 7.5
- `cls`: 0.5
- `dfl`: 1.5

## Data

The 50 test image pairs and their merged annotations are available on request from the corresponding authors.

## Citation

J. Zhao, M. Li, J. Lou, Y. Pang, X. Huang, J. Yang, Y. Ma. BiSSR-Net: A bimodal oriented object detection framework for rice seed-setting rate estimation. Submitted to Pattern Recognition, 2026.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
