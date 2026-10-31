"""ROS 2 detection node: sensor_msgs/Image -> vision_msgs/Detection2DArray."""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import (Detection2D, Detection2DArray,
                             ObjectHypothesisWithPose)
from ultralytics import YOLO


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector_node")
        self.declare_parameter("weights", "best.pt")
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("imgsz", 960)
        self.declare_parameter("device", "0")
        self.declare_parameter("image_topic", "/camera/image_raw")

        self.model = YOLO(self.get_parameter("weights").value)
        self.conf = self.get_parameter("conf").value
        self.imgsz = self.get_parameter("imgsz").value
        self.device = self.get_parameter("device").value
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, self.get_parameter("image_topic").value,
            self.on_image, qos_profile_sensor_data)
        self.pub = self.create_publisher(Detection2DArray, "/detections", 10)
        self.get_logger().info("detector ready")

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]

        out = Detection2DArray()
        out.header = msg.header
        for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.cpu().numpy(),
                                          res.boxes.conf.cpu().numpy(),
                                          res.boxes.cls.cpu().numpy()):
            d = Detection2D()
            d.header = msg.header
            d.bbox.center.position.x = float((x1 + x2) / 2)
            d.bbox.center.position.y = float((y1 + y2) / 2)
            d.bbox.size_x = float(x2 - x1)
            d.bbox.size_y = float(y2 - y1)
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id = res.names[int(k)]
            h.hypothesis.score = float(c)
            d.results.append(h)
            out.detections.append(d)
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(DetectorNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
