#!/usr/bin/env python3
"""
Image crop node — removes a white-line artifact from the top of the Gazebo
camera frame before sending the image to VIO (VINS).

Subscribes:  /camera/image_raw        sensor_msgs/Image  (640x480)
Publishes:   /camera/image_cropped    sensor_msgs/Image  (640x465)

Crop: TOP_ROWS rows removed from the top.
After cropping, update cam0_pinhole.yaml:
  image_height: 465
  cy:           225.0   (= 240.0 - 15)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

TOP_ROWS = 15   # pixels to remove from the top


class ImageCropNode(Node):

    def __init__(self):
        super().__init__('image_crop_node')
        self._pub = self.create_publisher(Image, '/camera/image_cropped', 10)
        self.create_subscription(Image, '/camera/image_raw', self._cb, 10)
        self.get_logger().info(f'Image crop node started — removing top {TOP_ROWS} rows')

    def _cb(self, msg: Image):
        # Each row = width * bytes_per_pixel
        bytes_per_pixel = len(msg.data) // (msg.height * msg.width)
        row_bytes = msg.width * bytes_per_pixel
        offset = TOP_ROWS * row_bytes

        out = Image()
        out.header = msg.header
        out.encoding = msg.encoding
        out.width = msg.width
        out.height = msg.height - TOP_ROWS
        out.step = msg.step
        out.is_bigendian = msg.is_bigendian
        out.data = msg.data[offset:]

        self._pub.publish(out)


def main():
    rclpy.init()
    node = ImageCropNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
