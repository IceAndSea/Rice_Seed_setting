"""Train BiSSR-Net on paired transmissive and reflective images."""

from ultralytics import YOLO

# BiSSR-Net always receives a paired image, so it uses the plain fusion block
# from DEA_ori.py. train_mmssr.py instead uses DEA.py, which can switch to a
# single-modality branch. Patching the class here lets both scripts run from
# the same repository without manually swapping files.
import ultralytics.nn.tasks as _nn_tasks
from ultralytics.nn.modules.layers.DEA_ori import DEA as _DEAPlainFusion

_nn_tasks.DEA = _DEAPlainFusion

# ─── User config ───────────────────────────────────────────────
MODEL_CFG = "ultralytics/cfg/models/fuse/DEYOLOseed12.yaml"  # BiSSR-Net model YAML (Setting 7, Supplementary Section S15)
DATA_YAML = "ultralytics/cfg/datasets/obb-seed.yaml"           # Dataset YAML
IMG_SZ    = 1024
BATCH     = 32
EPOCHS    = 300
DEVICE    = "0"         # "0" | "0,1,2,3" | "cpu"
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load pretrained tensors with matching names and shapes.
    model = YOLO(MODEL_CFG).load("yolo12n.pt")
    model.train(
        data=DATA_YAML,
        ch=6,
        imgsz=IMG_SZ,
        epochs=EPOCHS,
        batch=BATCH,
        workers=0,
        device=DEVICE,
        optimizer="SGD",
        patience=40,
        amp=False,
        close_mosaic=0,  # keep mosaic augmentation until the last epoch
        cache="disk",
        project="runs/train_bissr",
        name="bissr_net",
        resume=False,
        fraction=1.0,
        # BiSSR-Net always sees a full image pair; "paired" disables RG-PMMT
        # random single-modality masking (see base.py apply_modality_dropout).
        pmmt_schedule="paired",
    )
