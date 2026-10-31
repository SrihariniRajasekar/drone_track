"""Constant-velocity Kalman filter + ByteTrack association. Pure numpy/scipy, no ROS.

Boxes are xyxy. Kalman state: [cx, cy, a, h, vcx, vcy, va, vh]  (a = w/h).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def xyxy_to_z(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])


def x_to_xyxy(x):
    cx, cy, a, h = x[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class KalmanBoxFilter:
    def __init__(self, box):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)            # dt = 1 frame
        self.H = np.eye(4, 8)
        z = xyxy_to_z(box)
        self.x = np.concatenate([z, np.zeros(4)])
        h = z[3]
        # noise scaled with box height, as in ByteTrack/SORT variants
        self._std_p, self._std_v = 1.0 / 20, 1.0 / 160
        self.P = np.diag(np.square([2 * self._std_p * h] * 2 + [1e-1] + [2 * self._std_p * h]
                                   + [10 * self._std_v * h] * 2 + [1e-5] + [10 * self._std_v * h]))

    def predict(self):
        h = self.x[3]
        Q = np.diag(np.square([self._std_p * h] * 2 + [1e-2] + [self._std_p * h]
                              + [self._std_v * h] * 2 + [1e-5] + [self._std_v * h]))
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        return x_to_xyxy(self.x)

    def update(self, box):
        h = self.x[3]
        R = np.diag(np.square([self._std_p * h] * 2 + [1e-1] + [self._std_p * h]))
        z = xyxy_to_z(box)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P


def iou_matrix(a, b):
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def associate(tracks, dets, min_iou):
    """Hungarian matching on IoU. Returns matches, unmatched track idx, unmatched det idx."""
    if not tracks or not len(dets):
        return [], list(range(len(tracks))), list(range(len(dets)))
    iou = iou_matrix([t.box for t in tracks], [d[:4] for d in dets])
    r, c = linear_sum_assignment(-iou)
    matches = [(i, j) for i, j in zip(r, c) if iou[i, j] >= min_iou]
    mt = {i for i, _ in matches}; md = {j for _, j in matches}
    return (matches,
            [i for i in range(len(tracks)) if i not in mt],
            [j for j in range(len(dets)) if j not in md])


class Track:
    _next_id = 1

    def __init__(self, det):
        self.kf = KalmanBoxFilter(det[:4])
        self.box = np.asarray(det[:4], float)
        self.score, self.cls = det[4], int(det[5])
        self.id = Track._next_id; Track._next_id += 1
        self.hits, self.time_since_update = 1, 0

    def predict(self):
        self.box = self.kf.predict()
        self.time_since_update += 1

    def update(self, det):
        self.kf.update(det[:4])
        self.box = x_to_xyxy(self.kf.x)
        self.score, self.cls = det[4], int(det[5])
        self.hits += 1
        self.time_since_update = 0


class ByteTracker:
    def __init__(self, high_thresh=0.5, low_thresh=0.1, new_track_thresh=0.6,
                 match_iou=0.2, low_match_iou=0.5, max_age=30, min_hits=2):
        self.hi, self.lo, self.new_th = high_thresh, low_thresh, new_track_thresh
        self.m_iou, self.lm_iou = match_iou, low_match_iou
        self.max_age, self.min_hits = max_age, min_hits
        self.tracks, self.frame = [], 0
        Track._next_id = 1

    def update(self, dets):
        """dets: array (N,6) [x1,y1,x2,y2,score,cls]. Returns list of (id, box, score, cls)."""
        self.frame += 1
        dets = np.asarray(dets, float).reshape(-1, 6)
        high = dets[dets[:, 4] >= self.hi]
        low = dets[(dets[:, 4] >= self.lo) & (dets[:, 4] < self.hi)]

        for t in self.tracks:
            t.predict()

        # stage 1: all tracks (active + recently lost) vs high-confidence dets
        m, ut, uh = associate(self.tracks, high, self.m_iou)
        for i, j in m:
            self.tracks[i].update(high[j])

        # stage 2: still-unmatched tracks vs low-confidence dets (ByteTrack's key idea)
        rem = [self.tracks[i] for i in ut]
        m2, ut2, _ = associate(rem, low, self.lm_iou)
        for i, j in m2:
            rem[i].update(low[j])

        # new tracks from unmatched confident detections
        for j in uh:
            if high[j][4] >= self.new_th:
                self.tracks.append(Track(high[j]))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [(t.id, t.box.copy(), t.score, t.cls) for t in self.tracks
                if t.time_since_update == 0 and (t.hits >= self.min_hits or self.frame <= self.min_hits)]
