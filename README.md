# DroneTrack

Replay recorded drone data through ROS 2: detect and track objects in the video,
and separately estimate the drone's own pose and motion.

## Status
- [x] Stage 1 detection: YOLOv8s fine-tuned on VisDrone-DET (trained on Colab T4), ROS 2 node written
- [x] Stage 2 tracking: Kalman filter + ByteTrack core (synthetic test passed), ROS 2 node, MOTA/IDF1 eval script written
- [ ] Tracking evaluation on VisDrone-MOT (script ready, not yet run: hit a Windows memory error, see Known issues)
- [ ] Stage 3 visual odometry (C++/OpenCV, stereo, EuRoC)
- [ ] Stage 4 EKF fusion
- [ ] Stage 5 ONNX / TensorRT benchmark
- [ ] Stage 6 RViz2 visualization

## Results so far

Detection: YOLOv8s, 50 epochs, imgsz 960, trained on VisDrone2019-DET train (6471 images).

| Split | Images | mAP50 | mAP50-95 |
|---|---|---|---|
| Val (used to pick best.pt) | 548 | 0.493 | 0.295 |
| **Test-dev (held out)** | 1610 | **0.399** | **0.232** |

Strongest classes on test-dev: car (0.792 mAP50), bus (0.616). Weakest: people (0.207), bicycle (0.174), awning-tricycle (0.244).

Speed numbers are not final. The validation-time figures (7 to 13 ms/image on a T4) are noisy;
a proper warmed-up benchmark is planned for Stage 5.

Tracking (MOTA, IDF1), state estimation (ATE) and TensorRT speed: not yet measured.

## Datasets
| Purpose | Dataset | Notes |
|---|---|---|
| Detection + tracking | **VisDrone** (DET + MOT) | drone-shot, 10 classes, tracking IDs |
| State estimation (VO + EKF) | **EuRoC MAV** | stereo + IMU + ground truth, ROS bags |
| Single-dataset fallback | KITTI | stereo, LiDAR, IMU, tracking labels, but car footage |

EuRoC has no objects, so it is NOT used for detection.

## Repo layout
```
scripts/        setup_windows.ps1, train_detector.py, eval_detector.py, eval_tracking.py
src/
  dronetrack_detection/   ROS 2 YOLO node
  dronetrack_tracking/    ByteTrack + Kalman core (tracker.py) and ROS 2 node
  dronetrack_data/        (empty) rosbag player
  dronetrack_vo/          (empty) stereo visual odometry
```
Data, weights and run outputs are git-ignored; see below for how to get them.

## Reproduce

Windows setup (keeps everything on E:):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```

Train on Colab (T4 GPU):
```python
!pip install ultralytics
!python train_detector.py --data-dir /content/data --project /content/drive/MyDrive/dronetrack/runs --epochs 50
```
Copy `best.pt` to `weights/best.pt`. VisDrone-DET downloads automatically.
Download VisDrone2019-MOT-val from the VisDrone GitHub page into `data/`.

Evaluate tracking:
```powershell
python scripts\eval_tracking.py --weights weights\best.pt --root data\VisDrone2019-MOT-val\VisDrone2019-MOT-val --stride 1
```

## Known issues
- `motmetrics` calls `np.asfarray` (removed in NumPy 2); `eval_tracking.py` avoids it by computing IoU itself.
  Other incompatibilities in `motmetrics` are possible.
- The first tracking evaluation attempt failed with a Windows `MemoryError` while importing NumPy
  (system memory or page file exhausted), not a code error. Free RAM or move the page file to another drive.
- IoU-only association can swap IDs when two objects overlap (seen in a synthetic test); appearance
  features (DeepSORT-style) would address it.
- Stage 3 onward needs Linux (WSL2 or Ubuntu) for ROS 2; TensorRT needs an NVIDIA GPU (Colab T4 works).

## Next steps
1. Run `eval_tracking.py` and record MOTA / IDF1.
2. Stereo visual odometry on EuRoC (C++/OpenCV), evaluate ATE with `evo`.
3. EKF fusion of IMU and visual odometry.
4. ONNX export, TensorRT FP16 benchmark vs PyTorch.
5. RViz2 visualization.
