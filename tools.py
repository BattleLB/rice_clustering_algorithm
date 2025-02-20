
"""
    This file is mainly a utility class,
    which contains some useful functions commonly used in this project
"""

import sys
import time
import os
import shutil
import G_V
import math
import cv2
import numpy as np



class Tool:
    def __init__(self):
        self.GV = G_V.Global()   # The setting of global variables



    # According to the current running time of the program,
    # a folder will be created for the output of the results that need to be used in the future
    def CreatePathOnTime(self, ALGO_VER):
        # Get the current time
        time_now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        # Get the current path
        pwd = os.getcwd()

        path = pwd + '/' + ALGO_VER + time_now[2:] + '/'
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        # print('文件夹创建完成  ', path)
        return path

    # Calculates the coordinates of the center point of the detection box
    def calculate_center_points(self, leftPoints, rightPoints):
        # Judge the legitimacy
        if len(leftPoints) != 2:
            print('The length of the vertex in the upper left corner is invalid.')
            return
        if len(rightPoints) != 2:
            print('The length of the vertex in the upper right corner is invalid.')
            return

        xMid = int((leftPoints[self.GV.xIndex] + rightPoints[self.GV.xIndex]) / 2)
        yMid = int((leftPoints[self.GV.yIndex] + rightPoints[self.GV.yIndex]) / 2)
        XY = [xMid, yMid]
        return XY



    def euclidean_distance(self, point1, point2):
        """计算两个点之间的欧几里得距离。

        参数:
        point1: tuple, 第一个点的坐标 (x1, y1)
        point2: tuple, 第二个点的坐标 (x2, y2)

        返回:
        float, 欧几里得距离

        """
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance


    # The minimum value is selected and its subscript in the original array is returned
    def get_min_value(self, Array):

        length = len(Array)
        if length == 0:
            print('It is not valid for the array length to be empty.')
            return -1
        elif length == 1:
            return 0
        else:
            min_value = float('inf')
            min_index = -1
            for i in range(length):
                if Array[i] <= min_value:
                    min_value = Array[i]
                    min_index = i
            return min_index


    # A dashed line is drawn by changing the thickness of the line segment according to the number of times it disappears
    def cnt_dotted_line(self, start_point, end_point, img, count, color):
        # Define the start and end points of the dashed line
        thickness = 8 - count * self.GV.scaleFactor  # Line width

        # Draw a dotted line
        num_segments = 3  # Divide the line into 10 segments
        for i in range(num_segments):
            # Calculate the start and end points of each segment
            segment_start = (
                int(start_point[0] + (end_point[0] - start_point[0]) * (i / num_segments)),
                int(start_point[1] + (end_point[1] - start_point[1]) * (i / num_segments))
            )
            segment_end = (
                int(start_point[0] + (end_point[0] - start_point[0]) * ((i + 0.5) / num_segments)),
                int(start_point[1] + (end_point[1] - start_point[1]) * ((i + 0.5) / num_segments))
            )
            # Draw each segment
            cv2.line(img, segment_start, segment_end, color, thickness)

        return img
