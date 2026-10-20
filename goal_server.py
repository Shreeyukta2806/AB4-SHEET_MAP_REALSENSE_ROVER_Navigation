#!/usr/bin/env python3
"""
Goal Server — CEAM GoKart Project, MAHE Bengaluru
Flask HTTP server on rover that accepts goal poses from a laptop.
No SSH required — send goals with a simple curl command.

Port: 5000
Run:
    source ~/seg_venv/bin/activate
    python3 ~/goal_server.py

If port busy:
    fuser -k 5000/tcp

Laptop curl examples:
    curl http://<ROVER_IP>:5000/status
    curl -X POST http://<ROVER_IP>:5000/goal \
         -H 'Content-Type: application/json' \
         -d '{"x":1.0,"y":0.5,"yaw":0.0}'
    curl -X POST http://<ROVER_IP>:5000/waypoints \
         -H 'Content-Type: application/json' \
         -d '{"points":[{"x":1.0,"y":0.0},{"x":1.0,"y":1.0}]}'
    curl -X POST http://<ROVER_IP>:5000/stop
"""

import threading
import math
from flask import Flask, request, jsonify
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

app = Flask(__name__)
ros_node = None


class GoalSender(Node):
    def __init__(self):
        super().__init__('goal_server')
        self._client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.get_logger().info('Goal server ready')

    def send_goal(self, x, y, yaw=0.0):
        self._client.wait_for_server()
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)
        goal.pose = pose
        self._client.send_goal_async(goal)
        self.get_logger().info(f'Goal sent: x={x} y={y} yaw={yaw}')


@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'online', 'rover': 'rover1'})


@app.route('/goal', methods=['POST'])
def receive_goal():
    data = request.json
    ros_node.send_goal(
        data.get('x', 0.0),
        data.get('y', 0.0),
        data.get('yaw', 0.0)
    )
    return jsonify({'status': 'goal sent'})


@app.route('/waypoints', methods=['POST'])
def waypoints():
    """Send a sequence of waypoints one after another (3 s apart)."""
    points = request.json.get('points', [])

    def run():
        import time
        for pt in points:
            ros_node.send_goal(pt['x'], pt['y'], pt.get('yaw', 0.0))
            time.sleep(3)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({'status': 'waypoints sent', 'count': len(points)})


@app.route('/stop', methods=['POST'])
def stop():
    ros_node.send_goal(0.0, 0.0, 0.0)
    return jsonify({'status': 'stop sent'})


def spin_ros():
    rclpy.spin(ros_node)


if __name__ == '__main__':
    rclpy.init()
    ros_node = GoalSender()
    threading.Thread(target=spin_ros, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
