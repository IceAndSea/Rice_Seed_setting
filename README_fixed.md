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
- `imagesIR/`: transmissive images (same file names)

## Installation

- Python >= 3.8
- PyTorch

The repository includes the modified `ultralytics` package (YOLOv12 with the BiSSR-Net modules). Run all scripts from the repository root.

## Training

BiSSR-Net (Setting 7 in the paper). Set `MODEL_CFG` in `train_bissr.py` to `ultralytics/cfg/models/fuse/dea-seed-fuse.yaml`, then run:

```bash
python train_bissr.py
```

MMSSR-Net. Set `MODEL_CFG` as above and `SCHEDULE = "full_warmup_120"` in `train_mmssr.py` to use the four-stage schedule reported in the paper, then run:

```bash
python train_mmssr.py
```

`train_bissr.py` uses the settings of the paper: 1024 x 1024 inputs, batch size 32, SGD, FP32 and up to 300 epochs (patience 40). In `train_mmssr.py`, `IMG_SZ = 640` and `BATCH = 16` are example defaults.

## Full-image inference

Reflective and transmissive images of a pair are stored in the same directory and named `<sample>-1.<ext>` (reflective) and `<sample>-2.<ext>` (transmissive).

```bash
python val_bissr.py --weights path/to/best.pt --img_dir path/to/images --conf 0.2
python val_mmssr.py --weights path/to/best.pt --img_dir path/to/images --mode dual   # or trans / reflect
```

Inference uses 1024 x 1024 slices with 300-pixel overlap, the Edge Confidence Penalty (score x 0.6 within 32 pixels of an internal slice edge), IoF containment filtering (IoF > 0.85 and area ratio < 0.5), and slice-based NMS.

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
