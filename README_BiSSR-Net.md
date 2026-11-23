# BiSSR-Net

**BiSSR-Net**: Bimodal-oriented object detection network for rice seed-setting rate estimation.

## Overview

BiSSR-Net is a dual-stream fusion network that processes two complementary imaging modalities for oriented bounding box (OBB) object detection. The architecture integrates:

- **ModalitySelector**: Splits 6-channel input into two 3-channel streams
- **Dual-stream backbone**: Parameter-unshared YOLOv12n with A2C2f modules
- **CMFT-DEPA**: Cross-Modal Feature Transport + Dual-Enhancement Positional Alignment fusion module
- **OBB detection head**: Oriented bounding box prediction

## Input Format

The model accepts **6-channel images** with the following channel convention:

| Channels | Modality | Physical Meaning |
|----------|----------|-----------------|
| `[0:3]`  | Transmissive | Bottom illumination (下打光) |
| `[3:6]`  | Reflective | Top illumination (上打光) |

Image files should be organized as:
- `images/` — reflective (top-illumination) images
- `imagesT/` — transmissive (bottom-illumination) images (corresponding files with same names)

## Installation

### Dependencies

- Python >= 3.8
- PyTorch >= 1.8
- Ultralytics YOLOv11

### Setup

1. Clone or install Ultralytics YOLOv11:
   ```bash
   pip install ultralytics
   ```

2. Copy the project files into the Ultralytics source directory:
   ```bash
   # Copy data module
   cp data/base_ori.py <ultralytics_path>/data/base_ori.py

   # Copy fusion module
   cp nn/modules/layers/DEA_ori.py <ultralytics_path>/nn/modules/layers/DEA_ori.py

   # Copy model tasks
   cp nn/tasks_ori.py <ultralytics_path>/nn/tasks_ori.py
   ```

3. Update `__init__.py` files to expose new modules:
   - Add imports in `<ultralytics_path>/nn/modules/__init__.py` for `DEA`
   - Add imports in `<ultralytics_path>/data/__init__.py` for `BaseDataset`

## Usage

### Training

```python
from ultralytics import YOLO

model = YOLO("your_model.yaml")
model.train(
    data="your_dataset.yaml",
    epochs=300,
    imgsz=640,
    batch=16,
)
```

### Inference

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(source="path/to/images", imgsz=640)
```

## Architecture Details

### Fusion Module (DEA)

The DEA (Dual-Enhancement Alignment) block consists of:

1. **CMFT-DECA**: Cross-Modal Feature Transport with cross-attention between modalities
2. **DEPA**: Dual-Enhancement Positional Alignment with spatial attention

### Hyperparameters

Loss weighting is controlled through the training hyperparameters:
- `hyp.box`: Box loss weight (default: 7.5)
- `hyp.cls`: Classification loss weight (default: 0.5)
- `hyp.dfl`: DFL loss weight (default: 1.5)

## Citation

If you use BiSSR-Net in your research, please cite:

```bibtex
@article{bissrnet2026,
  title={BiSSR-Net: Bimodal-Oriented Object Detection Network for Rice Seed-Setting Rate Estimation},
  author={Your Name et al.},
  journal={TBD},
  year={2026}
}
```

## License

This project is released under the AGPL-3.0 License, consistent with the Ultralytics framework.
