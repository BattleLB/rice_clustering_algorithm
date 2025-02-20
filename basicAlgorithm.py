

"""
    This file mainly contains a basic class of the algorithm,
     which is used for the inheritance of subsequent algorithms
"""

from tools import Tool
from G_V import Global
from configure import Configure
import sys
import cv2



class BasicAlgorithm:
    def __init__(self, objDataPath):
        self.GV = Global()
        self.TOOL = Tool()
        self.CONFIGURE = Configure('power_shell')
        self.objDataPath = objDataPath  # Accepts the open path of the data file
        self.saveDataPath = self.TOOL.CreatePathOnTime(self.CONFIGURE.ALGO_VER) # The output path that accepts the final algorithm processing result
        self.nowNum = self.system()  # Get the parameters of the system-read image passed in by the command line
        self.img = []  # Information used to store the picture that is currently being processed
        self.line = []  # Used to store rows of data in the currently read data file



    def main(self):
        print(self.nowNum)


    # Used to determine if the starting number of an incoming image is legitimate
    def system(self):
        if len(sys.argv) == 2:
            startNum = int(sys.argv[1])
            if startNum % 16 != 0:
                print('wrong, have to be 16x...')
                sys.exit(0)
        else:
            startNum = 0
        return startNum

    # Judgment of data legitimacy
    def judgment(self):
        # Determine whether the data in the entire file has been read or not
        if not self.line:
            print('eof reached, exit...')
            return False
        else:
            return True

    # Get image information
    def get_image_information(self):
        self.picNameLeft = './ytsplit/left' + format(self.nowNum, '06d') + '.jpg'
        print('now read pic ', self.picNameLeft)
        self.img = cv2.imread(self.picNameLeft)


    # A simple legality judgment for a list
    def list_judgment(self, data):
        if len(data) == 0:
            print('The current incoming data length is empty, and no further operations are required.')
            return False
        else:
            return True







    def my_rectangle(self, centerPoints, color, thickness):
        x_center = centerPoints[0]
        y_center = centerPoints[1]

        x1 = int(x_center - int(self.GV.longEdgeLength / 2))
        y1 = int(y_center - int(self.GV.shortEdgeLength / 2))
        x2 = int(x_center + int(self.GV.longEdgeLength / 2))
        y2 = int(y_center + int(self.GV.shortEdgeLength / 2))
        cv2.rectangle(self.img, [x1 + self.GV.trimOffset, y1], [x2 + self.GV.trimOffset, y2], color, thickness)



    def my_dotted_rectangle(self, centerPoints, color):
        x_center = centerPoints[0]
        y_center = centerPoints[1]

        x1 = int(x_center - int(self.GV.longEdgeLength / 2))
        y1 = int(y_center - int(self.GV.shortEdgeLength / 2))
        x2 = int(x_center + int(self.GV.longEdgeLength / 2))
        y2 = int(y_center - int(self.GV.shortEdgeLength / 2))
        self.img = self.TOOL.cnt_dotted_line((x1 + self.GV.trimOffset, y1), (x2 + self.GV.trimOffset, y2), self.img,
                                             centerPoints[2], color)

        x1 = x1
        y1 = y1 + self.GV.shortEdgeLength
        x2 = x2
        y2 = y2 + self.GV.shortEdgeLength
        self.img = self.TOOL.cnt_dotted_line((x1 + self.GV.trimOffset, y1), (x2 + self.GV.trimOffset, y2), self.img,
                                             centerPoints[2], color)

        x1 = x1
        y1 = y1 - self.GV.shortEdgeLength
        x2 = x1
        y2 = y2
        self.img = self.TOOL.cnt_dotted_line((x1 + self.GV.trimOffset, y1), (x2 + self.GV.trimOffset, y2), self.img,
                                             centerPoints[2], color)

        x1 = x1 + self.GV.longEdgeLength
        y1 = y1
        x2 = x1
        y2 = y2
        self.img = self.TOOL.cnt_dotted_line((x1 + self.GV.trimOffset, y1), (x2 + self.GV.trimOffset, y2), self.img,
                                             centerPoints[2], color)


