"""Evaluate detector + ByteTrack on VisDrone-MOT: reports MOTA and IDF1.

Expected layout (VisDrone2019-MOT-val):
  <root>/sequences/<seq>/0000001.jpg ...
  <root>/annotations/<seq>.txt   rows: frame,id,x,y,w,h,score,category,trunc,occ
"""
import argparse, glob, os, sys, time
import numpy as np
import motmetrics as mm
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "dronetrack_tracking"))
from dronetrack_tracking.tracker import ByteTracker, iou_matrix  # noqa: E402

VALID = set(range(1, 11))  # categories 0 (ignored) and 11 (others) are dropped


def load_gt(path):
    """Returns {frame: [(id, [x1,y1,x2,y2]), ...]}"""
    gt = {}
    for row in np.loadtxt(path, delimiter=",", ndmin=2):
        f, tid, x, y, w, h, score, cat = row[:8]
        if score == 0 or int(cat) not in VALID:
            continue
        gt.setdefault(int(f), []).append((int(tid), [x, y, x + w, y + h]))
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--root", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame (speeds up CPU runs)")
    a = ap.parse_args()

    model = YOLO(a.weights)
    accs, names = [], []
    seqs = sorted(os.listdir(os.path.join(a.root, "sequences")))
    if a.max_seqs:
        seqs = seqs[:a.max_seqs]

    for seq in seqs:
        gt = load_gt(os.path.join(a.root, "annotations", seq + ".txt"))
        frames = sorted(glob.glob(os.path.join(a.root, "sequences", seq, "*.jpg")))
        tracker, acc = ByteTracker(), mm.MOTAccumulator(auto_id=True)
        t0 = time.time()
        for i, fp in enumerate(frames, 1):
            if (i - 1) % a.stride:
                continue
            r = model.predict(fp, imgsz=a.imgsz, conf=0.1, device=a.device, verbose=False, max_det=1000)[0]
            if len(r.boxes):
                dets = np.hstack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None],
                                  r.boxes.cls.cpu().numpy()[:, None]])
            else:
                dets = np.zeros((0, 6))
            out = tracker.update(dets)

            g = gt.get(i, [])
            gids = [t for t, _ in g]
            pids = [o[0] for o in out]
            if g and out:
                iou = iou_matrix([b for _, b in g], [o[1] for o in out])
                dist = 1.0 - iou
                dist[iou < 0.5] = np.nan          # no match allowed below IoU 0.5
            else:
                dist = np.full((len(g), len(out)), np.nan)
            acc.update(gids, pids, dist)
            if i % 50 == 1:
                print(f"  {seq} frame {i}/{len(frames)}  ({time.time() - t0:.0f}s)", flush=True)
        accs.append(acc); names.append(seq)
        print("done", seq, flush=True)

    summ = mm.metrics.create().compute_many(accs, names=names, metrics=["mota", "idf1", "num_switches"],
                                            generate_overall=True)
    print(summ)


if __name__ == "__main__":
    main()