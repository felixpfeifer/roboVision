import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from roboVision import RoboVision


class RoboVisionNode(Node):
    def __init__(self, topic_name_in, topic_name_out):
        super().__init__("Robo_Vision")
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic_name_in, self.image_callback, 50)
        self.image_pub = self.create_publisher(Image,topic_name_out,50)

        # RoboVision
        modelpath = "/share/roboVision/model/best.pt"
        self.vision = RoboVision(modelpath)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(e)
        # Call the vision system
        results = self.vision.detect(cv_image)
        # Publish the results
        self.publish_results(results)

    def publish_results(self, results):
        """
        Publish the results to a ROS2 topic with the robo Vision Interface Message 
        :param results: The results to publish
        """
        # Create a new image message
        image_msg = Image()
        # Convert the image to a ROS2 message
        image_msg = self.bridge.cv2_to_imgmsg(results, "bgr8")
        # Publish the image message
        self.image_pub.publish(image_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RoboVisionNode()
    rclpy.spin(node)
    rclpy.shutdown()