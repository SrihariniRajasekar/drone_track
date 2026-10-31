"""ROS 2 tracking node: /detections -> ByteTrack -> /tracks (Detection2DArray with persistent ids)."""
import numpy as np
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from .tracker import ByteTracker


class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")
        self.declare_parameter("max_age", 30)
        self.declare_parameter("min_hits", 2)
        self.tracker = ByteTracker(max_age=self.get_parameter("max_age").value,
                                   min_hits=self.get_parameter("min_hits").value)
        self.names, self.name_to_idx = [], {}
        self.sub = self.create_subscription(Detection2DArray, "/detections", self.on_dets, 10)
        self.pub = self.create_publisher(Detection2DArray, "/tracks", 10)

    def _cls_idx(self, name):
        if name not in self.name_to_idx:
            self.name_to_idx[name] = len(self.names); self.names.append(name)
        return self.name_to_idx[name]

    def on_dets(self, msg: Detection2DArray):
        rows = []
        for d in msg.detections:
            cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
            w, h = d.bbox.size_x, d.bbox.size_y
            r = d.results[0].hypothesis
            rows.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, r.score, self._cls_idx(r.class_id)])
        tracks = self.tracker.update(np.array(rows).reshape(-1, 6))

        out = Detection2DArray(); out.header = msg.header
        for tid, box, score, cls in tracks:
            d = Detection2D(); d.header = msg.header
            d.id = str(tid)
            d.bbox.center.position.x = float((box[0] + box[2]) / 2)
            d.bbox.center.position.y = float((box[1] + box[3]) / 2)
            d.bbox.size_x = float(box[2] - box[0]); d.bbox.size_y = float(box[3] - box[1])
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id = self.names[cls]; h.hypothesis.score = float(score)
            d.results.append(h); out.detections.append(d)
        self.pub.publish(out)


def main():
    rclpy.init(); rclpy.spin(TrackerNode()); rclpy.shutdown()
