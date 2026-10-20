#!/usr/bin/env python3
"""
BEV Map Node — CEAM GoKart Project, MAHE Bengaluru
Produces a Bird's Eye View of the track surface using
a fixed homography from the downward-facing RealSense D435I.
No depth data required — RGB only.

Camera: RealSense D435I, 640x480 @ 15fps, tilt ~62.6 deg downward
Publishes: /bev/image (sensor_msgs/Image, RGB8, 400x400)

Run:
    source ~/seg_venv/bin/activate
    python3 ~/bev_map_node.py

Foxglove: subscribe /bev/image, set Color mode → RGB
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
import numpy as np
import cv2


class BEVMapNode(Node):
    def __init__(self):
        super().__init__('bev_map_node')

        # BEST_EFFORT required — Foxglove image panels won't show RELIABLE topics
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)

        self.sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.cb, qos)
        self.pub = self.create_publisher(Image, '/bev/image', qos)

        # ── Homography calibration ─────────────────────────────────────────
        # src: 4 ground-plane corners in 640x480 camera image
        # Tune these points to match your camera mount / tilt angle.
        src = np.float32([
            [160, 480],   # bottom-left
            [480, 480],   # bottom-right
            [540, 280],   # top-right
            [100, 280],   # top-left
        ])

        self.OUT_W, self.OUT_H = 400, 400

        # dst: where those corners land in the 400x400 top-down canvas
        dst = np.float32([
            [50,  380],
            [350, 380],
            [350,  50],
            [50,   50],
        ])

        self.H_mat = cv2.getPerspectiveTransform(src, dst)
        self.get_logger().info('BEV map node ready — publishing /bev/image')

    def cb(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)

        # Warp to top-down view
        bev = cv2.warpPerspective(img, self.H_mat, (self.OUT_W, self.OUT_H))

        # Green dot = rover position (centre-bottom of BEV canvas)
        cv2.circle(bev, (self.OUT_W // 2, self.OUT_H - 20), 8, (0, 255, 0), -1)

        out = Image()
        out.header   = msg.header
        out.height   = self.OUT_H
        out.width    = self.OUT_W
        out.encoding = 'rgb8'
        out.step     = self.OUT_W * 3
        out.data     = bev.tobytes()
        self.pub.publish(out)


def main():
    rclpy.init()
    node = BEVMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
