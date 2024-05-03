"""
This module is responsible for the vision system of the robot.
It will be responsible for detecting the objects
@Author: Felix Pfeifer
@Version: 1.1
"""

import cv2
import numpy as np
import torch
import pandas as pd

import os
import sys

from ultralytics import YOLO


class RoboVision:
    """
    The RoboVision class is responsible for detecting objects in a video using a YOLO model.
    :param model_path: The path to the YOLO model file.
    :param headless: Whether to run in headless mode.
    """
    def __init__(self, model_path, headless=False):
        """
        Initialize the RoboVision object.

        Args:
            model_path (str): The path to the YOLO model file.
            headless (bool, optional): Whether to run in headless mode. Defaults to False.
        """
        self.model = None
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file does not exist")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise ValueError("Invalid YOLO model file") from e
        if not isinstance(headless, bool):
            raise TypeError("headless parameter must be a boolean")
        self.headless = headless

    def detect_video(self, video):
        """
        This function will detect the objects in the video
        :param video path to the video file
        :return: the detected objects with a Vector of the following format:
            [x, y, rotation ,class]
        """
        # Check if the video file exists
        if not os.path.exists(video):
            raise FileNotFoundError("Video file does not exist")

        # Read the video with the given path
        video_capture = cv2.VideoCapture(video)
        if not video_capture.isOpened():
            raise Exception("Failed to open video file")

        results = []
        try:
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break

                # Detect the objects in the frame
                if not self.headless:
                    self.b_mask = np.zeros(frame.shape[:2], np.uint8)

                frame_results = self.detect(frame)
                # Add the results to the list of results

                if not self.headless:
                    cv2.imshow("b_mask", self.b_mask)
                    cv2.imshow("frame", frame)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break

                results.append(frame_results)
        finally:
            cv2.destroyAllWindows()

        return results

    def detect(self, image):
        """
        This function will detect the objects in the image
        :param image: from OpenCV
        :return: the detected objects with a Vector of the following format:
            [x, y, rotation ,class]
        """
        results = self.model(image)
        result = results[0]
        return_results = []

        # Detect each object from the list of Results
        for i, mask in enumerate(result.masks.xy):
            cls = int(result.boxes.cls[i].item())
            name_class = result.names[cls]
            mask = mask.astype(int)
            # Write the mask to the binary image
            # Erzeuge eine Kontur aus den gerundeten Koordinaten
            mask_points = mask.reshape((-1, 1, 2))
            # Zeichne die Maske auf das binäre Bild
            if not self.headless:
                self.draw_frame(mask_points)
            min_rect = cv2.minAreaRect(mask_points)

            center = min_rect[0]

            alpha = min_rect[2]
            # Print to Console the Rotation angle in radians
            alpha = np.radians(alpha)
            print("Class of the Object", name_class)
            print("Rotation angle: ", alpha)
            # Print to Console the Coords
            print("Center coordinates: ", min_rect[0])
            return_results.append([name_class, center, alpha])
        return return_results

    def draw_frame(self, object_mask, color, thickness):
        """
        Draws a frame around the object specified by the object_mask.

        Args:
            object_mask: A binary mask representing the object.
            color: The color of the frame.
            thickness: The thickness of the frame.

        Returns:
            None
        """
        try:
            cv2.drawContours(self.b_mask, [object_mask], -1, color, thickness)
        except Exception as e:
            print("Error occurred in cv2.drawContours:", e)


if __name__ == '__main__':
    def main(headless=False):
        """
        The main function for detecting objects in a video using RoboVision.

        Args:
            headless (bool): When False, it displays in two windows the binary contour and the video
                             frame with the detected objects.
        """
        video_path = "../data/input/video/Test_Manuell_1.avi"
        model_file_path = "../data/model/yolo8seg_m/best.pt"
        robovision = RoboVision(model_file_path, headless=headless)
        robovision.detect_video(video_path)

    main(headless=False)



