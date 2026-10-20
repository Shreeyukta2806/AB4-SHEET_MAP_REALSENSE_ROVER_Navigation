import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
import numpy as np
import struct

OBSTACLE_IDS = {0, 1, 9, 14, 20}  # wall, building, fence, pole, traffic sign

class SegCostmapNode(Node):
    def __init__(self):
        super().__init__('seg_costmap_node')
        self.sub = self.create_subscription(Image, '/seg/labels', self.cb, 10)
        self.pub_costmap = self.create_publisher(OccupancyGrid, '/seg/costmap', 10)
        self.pub_cloud  = self.create_publisher(PointCloud2,   '/seg/obstacles', 10)

        # RealSense D435I 640x480 intrinsics (approximate)
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        self.assumed_depth = 1.5   # metres — flat projection assumption
        self.res = 0.05            # costmap resolution m/cell
        self.width = 60            # cells (3m)
        self.height = 60

        self.get_logger().info('SegCostmap node started')

    def cb(self, msg):
        labels = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        mask = np.isin(labels, list(OBSTACLE_IDS))

        # ---- OccupancyGrid (unchanged behaviour) ----
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = 'base_link'
        grid.info.resolution = self.res
        grid.info.width  = self.width
        grid.info.height = self.height
        grid.info.origin.position.x = -0.5
        grid.info.origin.position.y = -float(self.width * self.res) / 2.0
        data = [0] * (self.width * self.height)
        ys, xs = np.where(mask)
        for px, py in zip(xs, ys):
            xw = (px - self.cx) / self.fx * self.assumed_depth
            yw = (py - self.cy) / self.fy * self.assumed_depth
            ci = int(( xw + 0.5) / self.res)
            cj = int(( yw + float(self.width * self.res) / 2.0) / self.res)
            if 0 <= ci < self.width and 0 <= cj < self.height:
                data[cj * self.width + ci] = 100
        grid.data = data
        self.pub_costmap.publish(grid)

        # ---- PointCloud2 for ObstacleLayer ----
        points = []
        for px, py in zip(xs, ys):
            x3d = (px - self.cx) / self.fx * self.assumed_depth
            y3d = 0.0   # flat ground assumption
            z3d = self.assumed_depth
            points.append((x3d, y3d, z3d))

        if points:
            cloud = self._make_cloud(grid.header, points)
            self.pub_cloud.publish(cloud)

    def _make_cloud(self, header, points):
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        data = b''.join(struct.pack('fff', *p) for p in points)
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width  = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step   = 12
        msg.row_step     = 12 * len(points)
        msg.data         = data
        msg.is_dense     = True
        return msg

def main():
    rclpy.init()
    rclpy.spin(SegCostmapNode())

if __name__ == '__main__':
    main()
