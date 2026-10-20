# ~/simple_segmentation_toolkit/seg_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image as PILImage

# ADE20K class IDs relevant to navigation
FLOOR_IDS   = {3}        # floor
WALL_IDS    = {0, 1}     # wall, building
OBSTACLE_IDS = {19, 56}  # furniture, door

PALETTE = {
    'floor':    (0, 255, 0),    # green = navigable
    'wall':     (255, 0, 0),    # red   = blocked
    'obstacle': (255, 165, 0),  # orange
    'other':    (100, 100, 100) # grey
}

class SegNode(Node):
    def __init__(self):
        super().__init__('seg_node')
        self.get_logger().info("Loading SegFormer-b0 ADE20K...")
        
        self.processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
       ).eval()
        
        self.get_logger().info("Model loaded ✅")
        
        self.sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self.image_cb, 10
        )
        self.pub_vis  = self.create_publisher(Image, '/seg/vis', 10)
        self.pub_mask = self.create_publisher(Image, '/seg/labels', 10)

    def image_cb(self, msg):
        # ROS Image → numpy
        img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )
        if msg.encoding == 'bgr8':
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        pil_img = PILImage.fromarray(img_np)
        
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits  # (1, num_classes, H/4, W/4)
        seg = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(msg.height, msg.width)]
        )[0].cpu().numpy().astype(np.uint8)
        
        # colorize
        vis = np.zeros((msg.height, msg.width, 3), dtype=np.uint8)
        vis[:] = PALETTE['other']
        for fid in FLOOR_IDS:
            vis[seg == fid] = PALETTE['floor']
        for wid in WALL_IDS:
            vis[seg == wid] = PALETTE['wall']
        for oid in OBSTACLE_IDS:
            vis[seg == oid] = PALETTE['obstacle']
        
        # publish vis
        vis_msg = Image()
        vis_msg.header = msg.header
        vis_msg.height, vis_msg.width = msg.height, msg.width
        vis_msg.encoding = 'rgb8'
        vis_msg.step = msg.width * 3
        vis_msg.data = vis.tobytes()
        self.pub_vis.publish(vis_msg)
        
        # publish raw label map
        mask_msg = Image()
        mask_msg.header = msg.header
        mask_msg.height, mask_msg.width = msg.height, msg.width
        mask_msg.encoding = 'mono8'
        mask_msg.step = msg.width
        mask_msg.data = seg.tobytes()
        self.pub_mask.publish(mask_msg)

def main():
    rclpy.init()
    node = SegNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
