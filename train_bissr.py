"""BiSSR-Net training script.

Copy the modified files (tasks_ori.py, base_ori.py, DEA_ori.py) into the
ultralytics package before running, or place them directly in this project's
ultralytics/ directory so they override the originals.
"""

from ultralytics import YOLO

# ─── User config ───────────────────────────────────────────────
MODEL_CFG = "ultralytics/cfg/models/fuse/dea-seed-fuse.yaml"  # BiSSR-Net model YAML
DATA_YAML = "ultralytics/cfg/datasets/obb-seed.yaml"           # Dataset YAML
IMG_SZ    = 1024
BATCH     = 32
EPOCHS    = 300
DEVICE    = "0"         # "0" | "0,1,2,3" | "cpu"
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = YOLO(MODEL_CFG)
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
        cache="disk",
        project="runs/train_bissr",
        name="bissr_net",
        resume=False,
        fraction=1.0,
    )
