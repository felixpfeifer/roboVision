#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from pyueye import ueye
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np


class IdsCameraInterfaceNode(Node):
    """
    Class for an IDS Camera to get a live feed and publish it to a ROS2 Topic
    No Image Processing is taken
    """

    def __init__(self, topic_name):
        super().__init__('IDS_Camera_Interface')
        self.nRet = None
        self.get_logger().info('Starting IDS Camera')
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, topic_name, 20)
        self.hCam = ueye.HIDS(0)  # 0: first available camera;  1-254: The camera with the specified camera ID
        self.sInfo = ueye.SENSORINFO()
        self.cInfo = ueye.CAMINFO()
        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.int()
        self.rectAOI = ueye.IS_RECT()
        self.pitch = ueye.INT()
        self.nBitsPerPixel = ueye.INT(24)  # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
        self.channels = 3  # 3: channels for color mode(RGB); take 1 channel for monochrome
        self.m_nColorMode = ueye.INT(1)  # Y8/RGB16/RGB24/REG32
        self.bytes_per_pixel = int(self.nBitsPerPixel / 8)

    def open(self):
        self.initialize_camera()
        self.set_display_mode()
        self.set_color_mode()
        self.allocate_image_memory()
        self.activate_live_video_mode()
        self.enable_queue_mode()
        self.capture()
        self.get_logger().info("IDS Camera starting live feed")
    def initialize_camera(self):
        """
        Start the IDS Camera connected with USB
        """
        nRet = ueye.is_InitCamera(self.hCam, None)
        if nRet != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")

        nRet = ueye.is_GetCameraInfo(self.hCam, self.cInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")

        nRet = ueye.is_GetSensorInfo(self.hCam, self.sInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")

        nRet = ueye.is_ResetToDefault(self.hCam)
        if nRet != ueye.IS_SUCCESS:
            print("is_ResetToDefault ERROR")
        self.nRet = nRet

    def set_display_mode(self):
        nRet = ueye.is_SetDisplayMode(self.hCam, ueye.IS_SET_DM_DIB)
        if nRet != ueye.IS_SUCCESS:
            print("is_SetDisplayMode ERROR")

    def set_color_mode(self):
        if int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
            ueye.is_GetColorDepth(self.hCam, self.nBitsPerPixel, self.m_nColorMode)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_BAYER: ")
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.m_nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
            self.m_nColorMode = ueye.IS_CM_BGRA8_PACKED
            self.nBitsPerPixel = ueye.INT(32)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_CBYCRY: ")
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
            self.m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_MONOCHROME: ")
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        else:
            self.m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("else")

    def allocate_image_memory(self):
        nRet = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, self.rectAOI, ueye.sizeof(self.rectAOI))
        if nRet != ueye.IS_SUCCESS:
            print("is_AOI ERROR")

        width = self.rectAOI.s32Width
        height = self.rectAOI.s32Height

        print("Camera model:\t\t", self.sInfo.strSensorName.decode('utf-8'))
        print("Camera serial no.:\t", self.cInfo.SerNo.decode('utf-8'))
        print("Maximum image width:\t", width)
        print("Maximum image height:\t", height)
        print()

        nRet = ueye.is_AllocImageMem(self.hCam, width, height, self.nBitsPerPixel, self.pcImageMemory, self.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            nRet = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                nRet = ueye.is_SetColorMode(self.hCam, self.m_nColorMode)

    def activate_live_video_mode(self):
        nRet = ueye.is_CaptureVideo(self.hCam, ueye.IS_DONT_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

    def enable_queue_mode(self):
        nRet = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID, self.rectAOI.s32Width,
                                       self.rectAOI.s32Height, self.nBitsPerPixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")
        else:
            print("Press q to leave the program")

    def ready(self) -> bool:
        """
        Check if the camera is ready to capture an image.

        Returns:
            bool: True if the camera is ready, False otherwise.
        """
        return self.nRet == ueye.IS_SUCCESS and rclpy.ok()

    def capture(self):
        while self.ready():
            array = ueye.get_data(self.pcImageMemory, self.rectAOI.s32Width, self.rectAOI.s32Height, self.nBitsPerPixel,
                                  self.pitch, copy=False)
            frame = np.reshape(array, (self.rectAOI.s32Height.value, self.rectAOI.s32Width.value, self.bytes_per_pixel))
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            self.image_publisher(frame)
            
    def image_publisher(self, frame):
        img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.image_pub.publish(img_msg)

    def close(self):
        ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)
        ueye.is_ExitCamera(self.hCam)


if __name__ == '__main__':
    def main():
        rclpy.init()
        node = IdsCameraInterfaceNode("/camera")
        node.open()
        node.close()
        rclpy.shutdown()

    main()
