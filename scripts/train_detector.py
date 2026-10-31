"""Fine-tune YOLO on VisDrone (detection).

Local (E: drive):   python scripts/train_detector.py
Colab / Kaggle:     !python train_detector.py --data-dir /content/data --device 0
Afterwards copy runs/detect/visdrone_yolo/weights/best.pt to E:\\drone_track\\weights\\best.pt
"""
import argparse
from ultralytics import YOLO, settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=960)  # VisDrone objects are tiny
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--data-dir", default="E:/drone_track/data")  # keeps VisDrone off C:
    p.add_argument("--project", default="runs/detect")
    args = p.parse_args()

    settings.update({"datasets_dir": args.data_dir})  # VisDrone.yaml downloads here

    model = YOLO(args.model)
    model.train(
        data="VisDrone.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name="visdrone_yolo",
        exist_ok=True,
        workers=2,
    )


if __name__ == "__main__":
    main()
