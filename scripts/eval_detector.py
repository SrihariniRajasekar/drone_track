"""Report detection mAP on the VisDrone val split."""
import sys
from ultralytics import YOLO

weights = sys.argv[1] if len(sys.argv) > 1 else "runs/detect/visdrone_yolo/weights/best.pt"
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 960

m = YOLO(weights).val(data="VisDrone.yaml", imgsz=imgsz, split="val")
print(f"mAP50     : {m.box.map50:.4f}")
print(f"mAP50-95  : {m.box.map:.4f}")
print(f"precision : {m.box.mp:.4f}")
print(f"recall    : {m.box.mr:.4f}")
