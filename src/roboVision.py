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
    def __init__(self, headless=False):
        self.model = YOLO("../data/model/yolo8seg_m/best.pt")
        self.headless = headless
       

    def detect_video(self, video):
        """
        This function will detect the objects in the video
        :param video path to the video file
        :return: the detected objects with a Vector of the following format:
            [x, y, rotation ,class]
        """
        # Read the video with the given path
        video = cv2.VideoCapture(video)
        results = []
        while True:

            ret, frame = video.read()
            if not ret:
                break
            # Detect the objects in the frame

            if self.headless == False:
                self.b_mask = np.zeros(frame.shape[:2], np.uint8)

            frame_results = self.detect(frame)
            # Add the results to the list of results
            
            if self.headless == False:
                    cv2.imshow("b_mask", self.b_mask)
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        

            results.append(frame_results)
        
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
        for i in range(len(result.masks.xy)):
            cls = int(result.boxes.cls[i].item())
            name_class = result.names[cls]
            mask = result.masks.xy[i].astype(int)
            # Write the mask to the binary image
            # Erzeuge eine Kontur aus den gerundeten Koordinaten
            mask_points = mask.reshape((-1, 1, 2))
            # Zeichne die Maske auf das binäre Bild
            if(self.headless == False):
                self.draw_frame(mask_points)
            min_rect = cv2.minAreaRect(mask_points)
            
            center = min_rect[0] 

            alpha = min_rect[2]
            # Print to Console the Rotation angle in radians
            alpha = alpha * np.pi / 180
            print("Class of the Object",name_class)
            print("Rotation angle: ", alpha)
            # Print to Console the Coords
            print("Center coordinates: ", min_rect[0])
            return_results.append([name_class,center,alpha])
        return return_results
    
    def draw_frame(self,object_mask):
        cv2.drawContours(self.b_mask, [object_mask], -1, (255), 5)



def main():
    roboVision = RoboVision(headless=False)
    roboVision.detect_video("../data/input/video/Test_Manuell_1.avi")


if __name__ == "__main__":
    main()
