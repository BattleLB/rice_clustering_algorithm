



"""
    This file mainly contains the implementation of a clustering algorithm
"""
import copy
import cv2
from basicAlgorithm import BasicAlgorithm



class CLUAlgorithm(BasicAlgorithm):
    def __init__(self, objDataPath):
        super().__init__(objDataPath)

        self.allDetectionBoxes = []     # All the detection frame data for this image

        self.lowFlag = [0, 0, 0]   # A flag that records whether the underlying update has not yet started
        self.middleFlag = [0, 0, 0]

        self.lowCenterPoints = []     # The underlying detection frame data
        self.middleCenterPoints = []  # Detection frame data in the middle layer
        self.uppCenterPoints = []     # Detection frame data in the upper layer


        self.lowClusterCnt = [0, 0, 0]  # The clustering used to record the underlying has not been updated many times
        self.middleClusterCnt = [0, 0, 0]


        self.lowCluster = [[120, self.GV.layerLower], [240, self.GV.layerLower], [360, self.GV.layerLower]]        # Clustering results at the bottom
        self.middleCluster = [[130, self.GV.layerMiddle], [240, self.GV.layerMiddle], [350, self.GV.layerMiddle]]

        self.lowEndDistance = [0, 0]
    def main(self):



        # The first thing to do is to open the data file that you want to read next
        objData = open(self.objDataPath, 'r')

        while True:
            self.line = objData.readline()

            # Determine whether the data in the entire file has been read or not
            if not self.judgment():
                break

            # Get the data of the detection frame of the first image that has been read so far
            nextNum = int(self.line[:8])
            if nextNum < self.nowNum:
                continue

            # If the number of detection frames in this image has been read, the calculation will begin
            if nextNum > self.nowNum:
                # Get image information
                self.get_image_information()

                # Stratify the data obtained by the detection frame
                self.layering()

                # update low
                self.update_low()


                # update middle
                self.update_middle()
                '''感觉一些数值的设定还存在很多问题'''


                # draw rectangle
                self.draw_rectangle()

                # draw line
                self.draw_line()










                # Save the processed image
                # saveFn = self.saveDataPath + 'left' + format(self.nowNum, '06d') + '.jpg'
                # # print('write to ', saveFn)
                # cv2.imwrite(saveFn, self.img)



                # Reset this series of data to zero
                self.reset()
                self.nowNum = nextNum



            # Add vertex coordinates for the detection box
            self.allDetectionBoxes.append(
                    [int(self.line[8:12]), int(self.line[12:16]), int(self.line[16:20]), int(self.line[20:24])])

        objData.close()



    # Reset this series of data to zero
    def reset(self):
        self.allDetectionBoxes = []
        self.lowCenterPoints = []
        self.middleCenterPoints = []
        self.uppCenterPoints = []



    # Stratify the data obtained by the detection frame
    def layering(self):

        # Legitimacy judgment
        if not self.list_judgment(self.allDetectionBoxes):
            return
        else:

            # get index information
            xI = self.GV.xIndex
            yI = self.GV.yIndex
            lowY = self.GV.layerLower
            midY = self.GV.layerMiddle
            uppY = self.GV.layerUpper
            fSize = self.GV.layerFSize

            for i in range(len(self.allDetectionBoxes)):
                # Calculate the center coordinates of the detection frame
                centerPoints = self.TOOL.calculate_center_points(self.allDetectionBoxes[i][0:2], self.allDetectionBoxes[i][2:4])

                if (lowY - fSize) < centerPoints[yI] < (lowY + fSize):
                    self.lowCenterPoints.append(centerPoints)
                elif (midY - fSize) < centerPoints[yI] < (midY + fSize):
                    self.middleCenterPoints.append(centerPoints)
                elif (uppY - fSize) < centerPoints[yI] < (uppY + fSize):
                    self.uppCenterPoints.append(centerPoints)



    # update low
    def update_low(self):


        # Legitimacy judgment
        if not self.list_judgment(self.lowCenterPoints):

            # get index information
            left = self.GV.leftIndex
            middle = self.GV.middleIndex
            right = self.GV.rightIndex

            if self.lowFlag[left] != 0:
                self.lowClusterCnt[left] += 1
            if self.lowFlag[middle] != 0:
                self.lowClusterCnt[middle] += 1
            if self.lowFlag[right] != 0:
                self.lowClusterCnt[right] += 1


            return
        else:

            temp_lowCenterPoints = copy.deepcopy(self.lowCenterPoints)

            # get index information
            xI = self.GV.xIndex
            yI = self.GV.yIndex
            left = self.GV.leftIndex
            middle = self.GV.middleIndex
            right = self.GV.rightIndex


            if self.lowFlag[middle] == 0:
                # initial update middle
                temp_lowCenterPoints, self.lowCluster[middle] = self.initial_basic_update_low(temp_lowCenterPoints,
                                                                                              self.lowCluster[middle],
                                                                                              middle)
            else:
                # update middle
                temp_lowCenterPoints, self.lowCluster[middle] = self.basic_update_low(temp_lowCenterPoints,
                                                                                      self.lowCluster[middle],
                                                                                      middle)
                if self.lowFlag[left] == 0 and self.lowFlag[right] == 0:
                    # initial update left
                    temp_lowCenterPoints, self.lowCluster[left] = self.initial_basic_update_low(temp_lowCenterPoints,
                                                                                                self.lowCluster[left],
                                                                                                left)

                    # initial update right
                    temp_lowCenterPoints, self.lowCluster[right] = self.initial_basic_update_low(temp_lowCenterPoints,
                                                                                                 self.lowCluster[right],
                                                                                                 right)
                elif self.lowFlag[left] != 0 and self.lowFlag[right] == 0:
                    # update left
                    temp_lowCenterPoints, self.lowCluster[left] = self.basic_update_low(temp_lowCenterPoints,
                                                                                        self.lowCluster[left], left)

                    # initial update right
                    temp_lowCenterPoints, self.lowCluster[right] = self.initial_basic_update_low(temp_lowCenterPoints,
                                                                                                 self.lowCluster[right],
                                                                                                 right)

                elif self.lowFlag[left] == 0 and self.lowFlag[right] != 0:
                    # update right
                    temp_lowCenterPoints, self.lowCluster[right] = self.basic_update_low(temp_lowCenterPoints,
                                                                                         self.lowCluster[right], right)

                    # initial update left
                    temp_lowCenterPoints, self.lowCluster[left] = self.initial_basic_update_low(temp_lowCenterPoints,
                                                                                                self.lowCluster[left],
                                                                                                left)

                elif self.lowFlag[left] != 0 and self.lowFlag[right] !=0:
                    # update left
                    temp_lowCenterPoints, self.lowCluster[left] = self.basic_update_low(temp_lowCenterPoints,
                                                                                        self.lowCluster[left], left)

                    # update right
                    temp_lowCenterPoints, self.lowCluster[right] = self.basic_update_low(temp_lowCenterPoints,
                                                                                         self.lowCluster[right], right)




        # Let's see if the clustering in the middle exists after such a long logical process
        if self.lowFlag[middle] != 0:


            # Check whether the current clustering meets the threshold boundary, and continue to correct if it does not
            self.check_boundary_low()

            # calculate Distance between the two ends
            self.calculate_two_ends()
        else:
            self.lowEndDistance = [99999, 99999]





    # initial update low
    def initial_basic_update_low(self, temp, result, location):
        if not self.list_judgment(temp):
            return temp, result
        else:
            maxDistance = self.GV.lowInitMaxSize

            distance = []

            # calculate distance
            for i in range(len(temp)):
                distance.append(self.TOOL.euclidean_distance(temp[i], result))

            min_index = self.TOOL.get_min_value(distance)
            if distance[min_index] < maxDistance:
                result = copy.deepcopy(temp[min_index])
                self.lowFlag[location] = 1
                del temp[min_index]
                return temp, result

            return temp, result


    # update low
    def basic_update_low(self, temp, result, location):
        if not self.list_judgment(temp):
            self.lowClusterCnt[location] += 1
            return temp, result
        else:

            # get index information
            xI = self.GV.xIndex
            yI = self.GV.yIndex
            oldWeights = self.GV.oldWeights
            newWeights = self.GV.newWeights

            maxDistance = self.GV.lowMaxSize

            distance = []

            # calculate distance
            for i in range(len(temp)):
                distance.append(self.TOOL.euclidean_distance(temp[i], result))

            min_index = self.TOOL.get_min_value(distance)
            if distance[min_index] < maxDistance:
                result[xI] = oldWeights * result[xI] + newWeights * temp[min_index][xI]
                result[yI] = oldWeights * result[yI] + newWeights * temp[min_index][yI]
                self.lowClusterCnt[location] = 0
                del temp[min_index]
                return temp, result
            else:
                self.lowClusterCnt[location] += 1
                return temp, result



    # Check whether the current clustering meets the threshold boundary, and continue to correct if it does not
    def check_boundary_low(self):

        # get index information
        xI = self.GV.xIndex
        left = self.GV.leftIndex
        middle = self.GV.middleIndex
        right = self.GV.rightIndex

        leftBoundary = self.lowCluster[middle][xI] - self.GV.limitLowLeft
        rightBoundary = self.lowCluster[middle][xI] + self.GV.limitLowRight

        if self.lowCluster[left][xI] > leftBoundary:
            self.lowCluster[left][xI] = leftBoundary
        if self.lowCluster[right][xI] < rightBoundary:
            self.lowCluster[right][xI] = rightBoundary



    # calculate Distance between the two ends
    def calculate_two_ends(self):
        self.lowEndDistance[0] = self.TOOL.euclidean_distance(self.lowCluster[self.GV.leftIndex], self.lowCluster[self.GV.middleIndex])

        self.lowEndDistance[1] = self.TOOL.euclidean_distance(self.lowCluster[self.GV.middleIndex], self.lowCluster[self.GV.rightIndex])


    # update middle
    def update_middle(self):


        # Legitimacy judgment
        if not self.list_judgment(self.middleCenterPoints):
            # get index information
            left = self.GV.leftIndex
            middle = self.GV.middleIndex
            right = self.GV.rightIndex

            if self.middleFlag[left] != 0:
                self.middleClusterCnt[left] += 1
            if self.middleFlag[middle] != 0:
                self.middleClusterCnt[middle] += 1
            if self.middleFlag[right] != 0:
                self.middleClusterCnt[right] += 1

            return
        else:

            temp_middleCenterPoints = copy.deepcopy(self.middleCenterPoints)

            # get index information
            xI = self.GV.xIndex
            yI = self.GV.yIndex
            left = self.GV.leftIndex
            middle = self.GV.middleIndex
            right = self.GV.rightIndex

            # print(self.middleFlag)
            # When none of the three classes exist
            if self.middleFlag[left] == 0 and self.middleFlag[middle] == 0 and self.middleFlag[right] == 0:

                # initial update middle
                temp_middleCenterPoints, self.middleCluster[middle] = self.initial_middle_update_middle(temp_middleCenterPoints,
                                                                                              self.middleCluster[middle],
                                                                                              middle)

                # initial update left
                temp_middleCenterPoints, self.middleCluster[left] = self.initial_middle_update_left(temp_middleCenterPoints, self.middleCluster[left], left)

                # initial update right
                temp_middleCenterPoints, self.middleCluster[right] = self.initial_middle_update_right(
                    temp_middleCenterPoints, self.middleCluster[right], right)

            # When only intermediate clusters exist
            elif self.middleFlag[left] == 0 and self.middleFlag[middle] != 0 and self.middleFlag[right] == 0:
                # update middle
                temp_middleCenterPoints, self.middleCluster[middle] = self.basic_middle_update_middle(temp_middleCenterPoints, self.middleCluster[middle], middle)

                # initial update left
                temp_middleCenterPoints, self.middleCluster[left] = self.initial_middle_update_left(
                    temp_middleCenterPoints, self.middleCluster[left], left)

                # initial update right
                temp_middleCenterPoints, self.middleCluster[right] = self.initial_middle_update_right(
                    temp_middleCenterPoints, self.middleCluster[right], right)

            # only left
            elif self.middleFlag[left] != 0 and self.middleFlag[middle] == 0 and self.middleFlag[right] == 0:
                # update left
                temp_middleCenterPoints, self.middleCluster[left] = self.basic_middle_update_left(
                    temp_middleCenterPoints, self.middleCluster[left], left)

                # initial update middle
                temp_middleCenterPoints, self.middleCluster[middle] = self.initial_middle_update_middle(
                    temp_middleCenterPoints,
                    self.middleCluster[middle],
                    middle)

                # initial update right
                temp_middleCenterPoints, self.middleCluster[right] = self.initial_middle_update_right(
                    temp_middleCenterPoints, self.middleCluster[right], right)

            # only right
            elif self.middleFlag[left] == 0 and self.middleFlag[middle] == 0 and self.middleFlag[right] != 0:

                # update right
                temp_middleCenterPoints, self.middleCluster[right] = self.basic_middle_update_right(
                    temp_middleCenterPoints, self.middleCluster[right], right)

                # initial update middle
                temp_middleCenterPoints, self.middleCluster[middle] = self.initial_middle_update_middle(
                    temp_middleCenterPoints,
                    self.middleCluster[middle],
                    middle)

                # initial update left
                temp_middleCenterPoints, self.middleCluster[left] = self.initial_middle_update_left(
                    temp_middleCenterPoints, self.middleCluster[left], left)



            # It's just that the left side doesn't exist
            elif self.middleFlag[left] == 0 and self.middleFlag[middle] != 0 and self.middleFlag[right] != 0:

                # update middle
                temp_middleCenterPoints, self.middleCluster[middle] = self.basic_middle_update_middle(
                    temp_middleCenterPoints, self.middleCluster[middle], middle)

                # update right
                temp_middleCenterPoints, self.middleCluster[right] = self.basic_middle_update_right(
                    temp_middleCenterPoints, self.middleCluster[right], right)

                # initial update left
                temp_middleCenterPoints, self.middleCluster[left] = self.initial_middle_update_left(
                    temp_middleCenterPoints, self.middleCluster[left], left)


            elif self.middleFlag[left] != 0 and self.middleFlag[middle] == 0 and self.middleFlag[right] != 0:


                # update left
                temp_middleCenterPoints, self.middleCluster[left] = self.basic_middle_update_left(
                    temp_middleCenterPoints, self.middleCluster[left], left)


                # update right
                temp_middleCenterPoints, self.middleCluster[right] = self.basic_middle_update_right(
                    temp_middleCenterPoints, self.middleCluster[right], right)


                # initial update middle
                temp_middleCenterPoints, self.middleCluster[middle] = self.initial_middle_update_middle(
                    temp_middleCenterPoints,
                    self.middleCluster[middle],
                    middle)



            elif self.middleFlag[left] != 0 and self.middleFlag[middle] != 0 and self.middleFlag[right] == 0:

                # update middle
                temp_middleCenterPoints, self.middleCluster[middle] = self.basic_middle_update_middle(
                    temp_middleCenterPoints, self.middleCluster[middle], middle)

                # update left
                temp_middleCenterPoints, self.middleCluster[left] = self.basic_middle_update_left(
                    temp_middleCenterPoints, self.middleCluster[left], left)

                # initial update right
                temp_middleCenterPoints, self.middleCluster[right] = self.initial_middle_update_right(
                    temp_middleCenterPoints, self.middleCluster[right], right)

            elif self.middleFlag[left] != 0 and self.middleFlag[middle] != 0 and self.middleFlag[right] != 0:
                # update middle
                temp_middleCenterPoints, self.middleCluster[middle] = self.basic_middle_update_middle(
                    temp_middleCenterPoints, self.middleCluster[middle], middle)

                # update left
                temp_middleCenterPoints, self.middleCluster[left] = self.basic_middle_update_left(temp_middleCenterPoints, self.middleCluster[left], left)

                # update right
                temp_middleCenterPoints, self.middleCluster[right] = self.basic_middle_update_right(
                    temp_middleCenterPoints, self.middleCluster[right], right)

        # Let's see if the clustering in the middle exists after such a long logical process
        if self.middleFlag[middle] != 0:
            # Check whether the current clustering meets the threshold boundary, and continue to correct if it does not
            self.check_boundary_middle()




    # initial update middle
    def initial_middle_update_middle(self, temp, result, location):
        if not self.list_judgment(temp):
            return temp, result
        else:

            distance = []

            # calculate distance
            for i in range(len(temp)):
                distance.append(self.TOOL.euclidean_distance(temp[i], result))

            min_index = self.TOOL.get_min_value(distance)
            if distance[min_index] < 40:
                result = copy.deepcopy(temp[min_index])
                self.middleFlag[location] = 1
                del temp[min_index]
                return temp, result

            return temp, result



    # initial update left
    def initial_middle_update_left(self, temp, result, location):
        if not self.list_judgment(temp):
            return temp, result
        else:
            maxDistance = self.GV.middleInitMaxSize
            distance = []

            # calculate distance
            for i in range(len(temp)):
                distance.append(self.TOOL.euclidean_distance(temp[i], result))

            min_index = self.TOOL.get_min_value(distance)
            if distance[min_index] < maxDistance:
                temp_result = copy.deepcopy(temp[min_index])
                # calculate with middle distance
                temp_distance = self.TOOL.euclidean_distance(temp_result, self.middleCluster[self.GV.middleIndex])
                if (temp_distance + self.GV.deviation) < self.lowEndDistance[0]:
                    result = temp_result
                    self.middleFlag[location] = 1
                    del temp[min_index]
                    return temp, result
            return temp, result



    # initial update right
    def initial_middle_update_right(self, temp, result, location):
        if not self.list_judgment(temp):
            return temp, result
        else:
            maxDistance = self.GV.middleInitMaxSize
            distance = []

            # calculate distance
            for i in range(len(temp)):
                distance.append(self.TOOL.euclidean_distance(temp[i], result))

            min_index = self.TOOL.get_min_value(distance)

            if distance[min_index] < maxDistance:
                temp_result = copy.deepcopy(temp[min_index])
                # calculate with middle distance
                temp_distance = self.TOOL.euclidean_distance(temp_result, self.middleCluster[self.GV.middleIndex])
                if (temp_distance + self.GV.deviation) < self.lowEndDistance[1]:
                    result = temp_result
                    self.middleFlag[location] = 1
                    del temp[min_index]
                    return temp, result
            return temp, result

    # update middle
    def basic_middle_update_middle(self, temp, result, location):

        if not self.list_judgment(temp):
            self.middleClusterCnt[location] += 1
            return temp, result
        else:

            # get index information
            xI = self.GV.xIndex
            yI = self.GV.yIndex
            oldWeights = self.GV.oldWeights
            newWeights = self.GV.newWeights



            distance = []

            # calculate distance
            for i in range(len(temp)):
                distance.append(self.TOOL.euclidean_distance(temp[i], result))

            min_index = self.TOOL.get_min_value(distance)
            if distance[min_index] < 50:
                result[xI] = oldWeights * result[xI] + newWeights * temp[min_index][xI]
                result[yI] = oldWeights * result[yI] + newWeights * temp[min_index][yI]
                self.middleClusterCnt[location] = 0
                del temp[min_index]
                return temp, result
            else:
                self.middleClusterCnt[location] += 1
                return temp, result

    # update left
    def basic_middle_update_left(self, temp, result, location):

        if not self.list_judgment(temp):
            self.middleClusterCnt[location] += 1
            return temp, result
        else:

            # get index information
            xI = self.GV.xIndex
            yI = self.GV.yIndex
            oldWeights = self.GV.oldWeights
            newWeights = self.GV.newWeights

            maxDistance = self.GV.middleMaxSize

            distance = []

            # calculate distance
            for i in range(len(temp)):
                distance.append(self.TOOL.euclidean_distance(temp[i], result))

            min_index = self.TOOL.get_min_value(distance)

            if distance[min_index] < maxDistance:
                temp_x = oldWeights * result[xI] + newWeights * temp[min_index][xI]
                temp_y = oldWeights * result[yI] + newWeights * temp[min_index][yI]

                temp_distance = self.TOOL.euclidean_distance([temp_x, temp_y], self.middleCluster[self.GV.middleIndex])
                if (temp_distance + self.GV.deviation) < self.lowEndDistance[0]:
                    result = [temp_x, temp_y]
                    self.middleClusterCnt[location] = 0
                    del temp[min_index]
                    return temp, result
                else:
                    self.middleClusterCnt[location] += 1
                    return temp, result
            else:
                self.middleClusterCnt[location] += 1
                return temp, result


    # update right
    def basic_middle_update_right(self, temp, result, location):

        if not self.list_judgment(temp):
            self.middleClusterCnt[location] += 1
            return temp, result
        else:

            # get index information
            xI = self.GV.xIndex
            yI = self.GV.yIndex
            oldWeights = self.GV.oldWeights
            newWeights = self.GV.newWeights

            maxDistance = self.GV.middleMaxSize

            distance = []

            # calculate distance
            for i in range(len(temp)):
                distance.append(self.TOOL.euclidean_distance(temp[i], result))

            min_index = self.TOOL.get_min_value(distance)

            if distance[min_index] < maxDistance:
                temp_x = oldWeights * result[xI] + newWeights * temp[min_index][xI]
                temp_y = oldWeights * result[yI] + newWeights * temp[min_index][yI]

                temp_distance = self.TOOL.euclidean_distance([temp_x, temp_y],
                                                                 self.middleCluster[self.GV.middleIndex])

                if (temp_distance + self.GV.deviation) < self.lowEndDistance[1]:
                    result = [temp_x, temp_y]
                    self.middleClusterCnt[location] = 0
                    del temp[min_index]
                    return temp, result
                else:
                    self.middleClusterCnt[location] += 1
                    return temp, result
            else:
                self.middleClusterCnt[location] += 1
                return temp, result



    # Check whether the current clustering meets the threshold boundary, and continue to correct if it does not
    def check_boundary_middle(self):

        # get index information
        xI = self.GV.xIndex
        left = self.GV.leftIndex
        middle = self.GV.middleIndex
        right = self.GV.rightIndex

        leftBoundary = self.middleCluster[middle][xI] - self.GV.limitMiddleLeft
        rightBoundary = self.middleCluster[middle][xI] + self.GV.limitMiddleRight

        if self.middleCluster[left][xI] > leftBoundary:
            self.middleCluster[left][xI] = leftBoundary
        if self.middleCluster[right][xI] < rightBoundary:
            self.middleCluster[right][xI] = rightBoundary



    # draw rectangle
    def draw_rectangle(self):

        # get index information
        left = self.GV.leftIndex
        middle = self.GV.middleIndex
        right = self.GV.rightIndex

        # draw left rectangle
        if self.lowClusterCnt[left] == 0 and self.lowFlag[left] == 1:
            self.my_rectangle(self.lowCluster[left], self.GV.yellow, 3)
        elif 0 < self.lowClusterCnt[left] < 5 and self.lowFlag[left] == 1:
            centerPoints = copy.deepcopy(self.lowCluster[left])
            centerPoints.append(self.lowClusterCnt[left])
            self.my_dotted_rectangle(centerPoints, self.GV.yellow)

        if self.middleClusterCnt[left] == 0 and self.middleFlag[left] == 1:
            self.my_rectangle(self.middleCluster[left], self.GV.yellow, 3)
        elif 0 < self.middleClusterCnt[left] < 5 and self.middleFlag[left] == 1:
            centerPoints = copy.deepcopy(self.middleCluster[left])
            centerPoints.append(self.middleClusterCnt[left])
            self.my_dotted_rectangle(centerPoints, self.GV.yellow)


        # draw middle rectangle
        if self.lowClusterCnt[middle] == 0 and self.lowFlag[middle] == 1:
            self.my_rectangle(self.lowCluster[middle], self.GV.blue, 3)
        elif 0 < self.lowClusterCnt[middle] < 5 and self.lowFlag[middle] == 1:
            centerPoints = copy.deepcopy(self.lowCluster[middle])
            centerPoints.append(self.lowClusterCnt[middle])
            self.my_dotted_rectangle(centerPoints, self.GV.blue)
        if self.middleClusterCnt[middle] == 0 and self.middleFlag[middle] == 1:
            self.my_rectangle(self.middleCluster[middle], self.GV.blue, 3)
        elif 0 < self.middleClusterCnt[middle] < 5 and self.middleFlag[middle] == 1:
            centerPoints = copy.deepcopy(self.middleCluster[middle])
            centerPoints.append(self.middleClusterCnt[middle])
            self.my_dotted_rectangle(centerPoints, self.GV.blue)


        # draw right rectangle
        if self.lowClusterCnt[right] == 0 and self.lowFlag[right] == 1:
            self.my_rectangle(self.lowCluster[right], self.GV.brown, 3)
        elif 0 < self.lowClusterCnt[right] < 5 and self.lowFlag[right] == 1:
            centerPoints = copy.deepcopy(self.lowCluster[right])
            centerPoints.append(self.lowClusterCnt[right])
            self.my_dotted_rectangle(centerPoints, self.GV.brown)

        if self.middleClusterCnt[right] == 0 and self.middleFlag[right] == 1:
            self.my_rectangle(self.middleCluster[right], self.GV.brown, 3)
        elif 0 < self.middleClusterCnt[right] < 5 and self.middleFlag[right] == 1:
            centerPoints = copy.deepcopy(self.middleCluster[right])
            centerPoints.append(self.middleClusterCnt[right])
            self.my_dotted_rectangle(centerPoints, self.GV.brown)





    # draw line
    def draw_line(self):


        # get index information
        xI = self.GV.xIndex
        yI = self.GV.yIndex
        left = self.GV.leftIndex
        middle = self.GV.middleIndex
        right = self.GV.rightIndex





        if self.lowClusterCnt[left] >= 5:
            if self.lowFlag[middle] == 0:
                self.lowCluster[left] = [120, self.GV.layerLower]
                self.lowFlag[left] = 0
                self.lowClusterCnt[left] = 0
            elif self.lowFlag[middle] != 0:
                self.lowCluster[left] = [self.lowCluster[middle][xI]-self.GV.updateLimitLowLeft, self.GV.layerLower]
                self.lowFlag[left] = 0
                self.lowClusterCnt[left] = 0
        if self.lowClusterCnt[middle] >= 5:
            self.lowCluster[middle] = [240, self.GV.layerLower]
            self.lowFlag[middle] = 0
            self.lowClusterCnt[middle] = 0
            if self.lowFlag[left] == 0:
                self.lowCluster[left] = [120, self.GV.layerLower]
                self.lowFlag[left] = 0
                self.lowClusterCnt[left] = 0
            if self.lowFlag[right] == 0:
                self.lowCluster[right] = [360, self.GV.layerLower]
                self.lowFlag[right] = 0
                self.lowClusterCnt[right] = 0

        if self.lowClusterCnt[right] >= 5:
            if self.lowFlag[middle] == 0:
                self.lowCluster[right] = [360, self.GV.layerLower]
                self.lowFlag[right] = 0
                self.lowClusterCnt[right] = 0
            elif self.lowFlag[middle] != 0:
                self.lowCluster[right] = [self.lowCluster[middle][xI]+self.GV.updateLimitLowRight, self.GV.layerLower]
                self.lowFlag[right] = 0
                self.lowClusterCnt[right] = 0







        if self.middleClusterCnt[left] >= 5:
            if self.middleFlag[middle] == 0:
                self.middleCluster[left] = [130, self.GV.layerMiddle]
                self.middleFlag[left] = 0
                self.middleClusterCnt[left] = 0
            elif self.middleFlag[middle] != 0:
                self.middleCluster[left] = [self.middleCluster[middle][xI] - self.GV.updateLimitMiddleLeft, self.GV.layerMiddle]
                self.middleFlag[left] = 0
                self.middleClusterCnt[left] = 0
        if self.middleClusterCnt[middle] >= 5:
            self.middleCluster[middle] = [240, self.GV.layerMiddle]
            self.middleFlag[middle] = 0
            self.middleClusterCnt[middle] = 0
            if self.middleFlag[left] == 0:
                self.middleCluster[left] = [130, self.GV.layerMiddle]
                self.middleFlag[left] = 0
                self.middleClusterCnt[left] = 0
            if self.middleFlag[right] == 0:
                self.middleCluster[right] = [350, self.GV.layerMiddle]
                self.middleFlag[right] = 0
                self.middleClusterCnt[right] = 0



        if self.middleClusterCnt[right] >= 5:
            if self.middleFlag[middle] == 0:
                self.middleCluster[right] = [350, self.GV.layerMiddle]
                self.middleFlag[right] = 0
                self.middleClusterCnt[right] = 0
            elif self.middleFlag[middle] != 0:
                self.middleCluster[right] = [self.middleCluster[middle][xI] + self.GV.updateLimitMiddleRight, self.GV.layerMiddle]
                self.middleFlag[right] = 0
                self.middleClusterCnt[right] = 0
