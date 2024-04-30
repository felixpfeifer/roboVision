"""
This module is responsible for the vision system of the robot.
It will be responsible for detecting the objects
@Author: Felix Pfeifer
@Version: 1.0

"""

import cv2
import numpy as np
import torch
import pandas as pd

import os
import sys

from ultralytics import YOLO


class RoboVision:
    def __init__(self):
        self.model = YOLO("data/model/yolo8seg_m/best.pt")
    def detect(self, image):
        """
        This function will detect the objects in the image
        :param image: from OpenCV
        :return: the detected objects with a Vector of the following format:
            [x, y, rotation ,class]
        """
        results = self.model(image)

        # Detect each object from the list of Results
        for obj in results[0]:
            # Get the position of the object
            position = self.find_positions(obj)
            # Add the class to the position
            position.append(obj[5])
            # Add the position to the list of results
            results.append(position)
        return results

    def find_positions(self, object):
        """
        This function will find the position of the object
        """
        contour = object.reshape((-1, 1, 2))
        # Find the minimum area rectangle
        rect = cv2.minAreaRect(contour)
        # Get the rotation
        rotation = rect[2]
        # Get the Center of the object
        center = rect[0]

        position = [center[0], center[1], rotation]
        return position


def main():
    roboVision = RoboVision()



if __name__ == "__main__":
    main()
