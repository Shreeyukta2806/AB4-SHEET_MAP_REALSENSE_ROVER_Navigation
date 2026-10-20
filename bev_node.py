#!/usr/bin/env python3
"""
BEV Node (Depth-based) — CEAM GoKart Project, MAHE Bengaluru
REFERENCE ONLY — not used in production.

This was the original depth-based BEV approach. It produced mostly-black
images because the D435I on USB 2.1 cannot get valid depth for near-ground
pixels at 15fps. Replaced by bev_map_node.py (RGB homography).

Kept for reference in case a proper USB 3.0 setup or a different depth
camera is used in the future.

Publishes:
  /bev/image   (sensor_msgs/Image, bgr8)
  /bev/costmap (nav_msgs/OccupancyGrid)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import numpy as np
import cv2


class BEVNode(Node):
    def __init__(self):
        super().__init__('bev_node')
        self.bridge = CvBridge()
        self.latest_rgb         = None
        self.latest_depth       = None
        self.latest_rgb_stamp   = None
        self.latest_depth_stamp = None
        self.last_proc_time     = 0.0
        self.frame_count        = 0

        # Camera intrinsics (D435I at 320x240 after resize)
        self.cam_height   = 0.147       # metres above ground
        self.cam_tilt_deg = 62.6        # degrees downward from horizontal
        self.fx, self.fy  = 303.787, 303.759
        self.cx, self.cy  = 161.763, 124.962

        # BEV grid params
        self.bev_res    = 0.005   # metres per pixel
        self.bev_width  = 300     # pixels
        self.bev_height = 200     # pixels

        sub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)

        pub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)

        self.create_subscription(Image,
            '/camera/camera/color/image_raw', self.rgb_cb, sub_qos)
        self.create_subscription(Image,
            '/camera/camera/aligned_depth_to_color/image_raw', self.depth_cb, sub_qos)

        self.pub_img = self.create_publisher(Image,         '/bev/image',   pub_qos)
        self.pub_bev = self.create_publisher(OccupancyGrid, '/bev/costmap', pub_qos)
        self.get_logger().info('BEV node ready (depth-based — reference only)')

    def rgb_cb(self, msg):
        self.latest_rgb = msg
        self.latest_rgb_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.try_process()

    def depth_cb(self, msg):
        self.latest_depth = msg
        self.latest_depth_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.try_process()

    def try_process(self):
        if self.latest_rgb is None or self.latest_depth is None:
            return
        if abs(self.latest_rgb_stamp - self.latest_depth_stamp) > 0.5:
            return
        now = self.latest_rgb_stamp
        if now - self.last_proc_time < 0.2:
            return
        self.last_proc_time = now
        self.process(self.latest_rgb, self.latest_depth)

    def process(self, rgb_msg, depth_msg):
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rgb = cv2.resize(rgb, (320, 240))

        depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(
                    depth_msg.height, depth_msg.width).astype(np.float32) / 1000.0
        depth = cv2.resize(depth, (320, 240))

        h, w   = depth.shape
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))

        Z     = depth
        valid = (Z > 0.05) & (Z < 4.0)
        X_c   = (uu - self.cx) * Z / self.fx
        Y_c   = (vv - self.cy) * Z / self.fy

        tilt   = np.radians(self.cam_tilt_deg)
        ct, st = np.cos(tilt), np.sin(tilt)
        Y_w    = Y_c * ct - Z * st + self.cam_height
        X_w    = X_c
        Z_w    = Y_c * st + Z * ct

        gnd = (valid
               & (np.abs(Y_w) < 0.30)
               & (Z_w > 0.01)
               & (Z_w < self.bev_height * self.bev_res))

        bev = np.full((self.bev_height, self.bev_width, 3), 30, dtype=np.uint8)
        bx  = ((X_w + self.bev_width * self.bev_res / 2) / self.bev_res).astype(int)
        by  = (self.bev_height - 1 - (Z_w / self.bev_res)).astype(int)
        ok  = (gnd
               & (bx >= 0) & (bx < self.bev_width)
               & (by >= 0) & (by < self.bev_height))
        bev[by[ok], bx[ok]] = rgb[vv[ok], uu[ok]]

        hsv         = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        white_mask  = cv2.inRange(hsv, ( 0,  0, 170), (180,  45, 255))
        yellow_mask = cv2.inRange(hsv, (15, 80,  80), ( 40, 255, 255))
        projected   = (bev[:,:,0] != 30) | (bev[:,:,1] != 30) | (bev[:,:,2] != 30)

        vis = bev.copy()
        vis[white_mask  > 0] = (255, 255, 255)
        vis[yellow_mask > 0] = (  0, 200, 255)
        cv2.circle(vis, (self.bev_width//2, self.bev_height-5), 5, (0, 255, 0), -1)

        img_msg                 = Image()
        img_msg.header.stamp    = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'base_link'
        img_msg.height          = self.bev_height
        img_msg.width           = self.bev_width
        img_msg.encoding        = 'bgr8'
        img_msg.step            = self.bev_width * 3
        img_msg.data            = np.ascontiguousarray(vis).tobytes()
        self.pub_img.publish(img_msg)

        occ = np.full((self.bev_height, self.bev_width), -1, dtype=np.int8)
        occ[projected]       =   0
        occ[white_mask  > 0] = 100
        occ[yellow_mask > 0] =  50

        grid                         = OccupancyGrid()
        grid.header.stamp            = self.get_clock().now().to_msg()
        grid.header.frame_id         = 'base_link'
        grid.info.resolution         = self.bev_res
        grid.info.width              = self.bev_width
        grid.info.height             = self.bev_height
        grid.info.origin.position.x  = -(self.bev_width * self.bev_res / 2)
        grid.info.origin.position.y  = 0.0
        grid.data                    = occ.flatten().tolist()
        self.pub_bev.publish(grid)

        self.frame_count += 1
        if self.frame_count % 10 == 0:
            self.get_logger().info(f'BEV frames published: {self.frame_count}')


def main():
    rclpy.init()
    rclpy.spin(BEVNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
