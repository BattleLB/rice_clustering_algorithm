


from clusteringAlgorithm import CLUAlgorithm
import copy
import cv2



class CLUAlgorithm2(CLUAlgorithm):

    def __init__(self, objDataPath):
        super().__init__(objDataPath)


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

                # initial low and middle
                self.research_middle()


                # update low
                self.update_low()

                # update middle
                self.update_middle()



                # draw rectangle
                self.draw_rectangle()



                """
                    在此处加上一个对中心聚类的绝对阈值的判断，若本次更新后的中心聚类越过了绝对阈值，
                    则对数据全部清零，重新处理
                """



                # draw line
                self.draw_line()









                # Save the processed image
                saveFn = self.saveDataPath + 'left' + format(self.nowNum, '06d') + '.jpg'
                # print('write to ', saveFn)
                cv2.imwrite(saveFn, self.img)

                # Reset this series of data to zero
                self.reset()
                self.nowNum = nextNum

            # Add vertex coordinates for the detection box
            self.allDetectionBoxes.append(
                [int(self.line[8:12]), int(self.line[12:16]), int(self.line[16:20]), int(self.line[20:24])])

        objData.close()




    # initial low and middle
    def research_middle(self):

        # get index information
        middle = self.GV.middleIndex
        xI = self.GV.xIndex

        if self.lowFlag[middle] == 0:
            if self.middleFlag[middle] != 0:
                temp_lowCenterPoints = copy.deepcopy(self.lowCenterPoints)

                self.lowCluster[middle][xI] = copy.deepcopy(self.middleCluster[middle][xI])

                # initial low middle
                temp_lowCenterPoints, self.lowCluster[middle] = self.initial_basic_update_low(temp_lowCenterPoints,
                                                                                              self.lowCluster[middle],
                                                                                              middle)

                self.lowCenterPoints = copy.deepcopy(temp_lowCenterPoints)

            else:

                temp_lowCenterPoints = copy.deepcopy(self.lowCenterPoints)

                # initial low middle
                temp_lowCenterPoints, self.lowCluster[middle] = self.initial_basic_update_low(temp_lowCenterPoints, self.lowCluster[middle], middle)

                self.lowCenterPoints = copy.deepcopy(temp_lowCenterPoints)

        """
            如果选到了下层的中心的合法框，要结合选定的下层的合法框的位置，去搜寻上层的框
        """


        if self.middleFlag[middle] == 0:
            if self.lowFlag[middle] != 0:
                temp_middleCneterPoints = copy.deepcopy(self.middleCenterPoints)

                self.middleCluster[middle][xI] = copy.deepcopy(self.lowCluster[middle][xI])

                # initial middle  middle
                temp_middleCneterPoints, self.middleCluster[middle] = self.initial_middle_update_middle(
                    temp_middleCneterPoints, self.middleCluster[middle], middle)

                self.middleCenterPoints = copy.deepcopy(self.middleCenterPoints)


            else:
                temp_middleCneterPoints = copy.deepcopy(self.middleCenterPoints)

                # initial middle  middle
                temp_middleCneterPoints, self.middleCluster[middle] = self.initial_middle_update_middle(temp_middleCneterPoints, self.middleCluster[middle], middle)

                self.middleCenterPoints = copy.deepcopy(self.middleCenterPoints)




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

            # get index information
            left = self.GV.leftIndex
            middle = self.GV.middleIndex
            right = self.GV.rightIndex

            temp_lowCenterPoints = copy.deepcopy(self.lowCenterPoints)


            if self.lowFlag[middle] == 0:

                if self.lowFlag[left] == 0:
                    self.lowCluster[left] = [120, self.GV.layerLower]
                    self.lowFlag[left] = 0
                    self.lowClusterCnt[left] = 0
                elif self.lowFlag[left] != 0:
                    # update left
                    temp_lowCenterPoints, self.lowCluster[left] = self.basic_update_low(temp_lowCenterPoints,
                                                                                        self.lowCluster[left], left)


                if self.lowFlag[right] == 0:
                    self.lowCluster[right] = [360, self.GV.layerLower]
                    self.lowFlag[right] = 0
                    self.lowClusterCnt[right] = 0
                elif self.lowFlag[right] != 0:

                    # update right
                    temp_lowCenterPoints, self.lowCluster[right] = self.basic_update_low(temp_lowCenterPoints,
                                                                                         self.lowCluster[right], right)



            if self.lowFlag[middle] != 0 and self.middleFlag[middle] != 0:
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

                elif self.lowFlag[left] != 0 and self.lowFlag[right] != 0:
                    # update left
                    temp_lowCenterPoints, self.lowCluster[left] = self.basic_update_low(temp_lowCenterPoints,
                                                                                        self.lowCluster[left], left)

                    # update right
                    temp_lowCenterPoints, self.lowCluster[right] = self.basic_update_low(temp_lowCenterPoints,
                                                                                         self.lowCluster[right], right)

            elif self.lowFlag[middle] != 0 and self.middleFlag[middle] == 0:
                # update middle
                temp_lowCenterPoints, self.lowCluster[middle] = self.basic_update_low(temp_lowCenterPoints,
                                                                                      self.lowCluster[middle],
                                                                                      middle)

                if self.lowFlag[left] == 0:
                    temp_lowCenterPoints, self.lowCluster[left] = self.initial_basic_update_low(temp_lowCenterPoints, self.lowCluster[left], left)
                elif self.lowFlag[left] != 0:
                    temp_lowCenterPoints, self.lowCluster[left] = self.basic_update_low(temp_lowCenterPoints, self.lowCluster[left], left)

                if self.lowFlag[right] == 0:
                    temp_lowCenterPoints, self.lowCluster[right] = self.initial_basic_update_low(temp_lowCenterPoints, self.lowCluster[right], right)
                elif self.lowFlag[right] != 0:
                    temp_lowCenterPoints, self.lowCluster[right] = self.basic_update_low(temp_lowCenterPoints, self.lowCluster[right], right)



        # Let's see if the clustering in the middle exists after such a long logical process
        if self.lowFlag[middle] != 0:


            # Check whether the current clustering meets the threshold boundary, and continue to correct if it does not
            self.check_boundary_low()

            # calculate Distance between the two ends
            self.calculate_two_ends()
        else:
            self.lowEndDistance = [99999, 99999]



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
            left = self.GV.leftIndex
            middle = self.GV.middleIndex
            right = self.GV.rightIndex




            if self.middleFlag[middle] == 0:

                if self.middleFlag[left] == 0:
                    self.middleCluster[left] = [130, self.GV.layerMiddle]
                    self.middleFlag[left] = 0
                    self.middleClusterCnt[left] = 0
                elif self.middleFlag[left] != 0:
                    temp_middleCenterPoints, self.middleCluster[left] = self.basic_middle_update_left(
                        temp_middleCenterPoints, self.middleCluster[left], left)

                if self.middleFlag[right] == 0:
                    self.middleCluster[right] = [350, self.GV.layerMiddle]
                    self.middleFlag[right] = 0
                    self.middleClusterCnt[right] = 0
                elif self.middleFlag[right] != 0:
                    # update middle right
                    temp_middleCenterPoints, self.middleCluster[right] = self.basic_middle_update_right(
                        temp_middleCenterPoints, self.middleCluster[right], right)


            if self.lowFlag[middle] == 0 and self.middleFlag[middle] != 0:

                temp_middleCenterPoints, self.middleCluster[middle] = self.basic_middle_update_middle(temp_middleCenterPoints, self.middleCluster[middle], middle)

                if self.middleFlag[left] == 0:
                    temp_middleCenterPoints, self.middleCluster[left] = self.initial_middle_update_left(temp_middleCenterPoints, self.middleCluster[left], left)
                elif self.middleFlag[left] != 0:
                    temp_middleCenterPoints, self.middleCluster[left] = self.basic_middle_update_left(temp_middleCenterPoints, self.middleCluster[left], left)

                if self.middleFlag[right] == 0:
                    temp_middleCenterPoints, self.middleCluster[right] = self.initial_middle_update_right(temp_middleCenterPoints, self.middleCluster[right], right)
                elif self.middleFlag[right] != 0:
                    temp_middleCenterPoints, self.middleCluster[right] = self.basic_middle_update_right(temp_middleCenterPoints, self.middleCluster[right], right)

            elif self.lowFlag[middle] != 0 and self.middleFlag[middle] != 0:



                temp_middleCenterPoints, self.middleCluster[middle] = self.basic_middle_update_middle(
                    temp_middleCenterPoints, self.middleCluster[middle], middle)

                if self.middleFlag[left] == 0 and self.middleFlag[right] == 0:

                    # initial update middle left
                    temp_middleCenterPoints, self.middleCluster[left] = self.initial_middle_update_left(temp_middleCenterPoints, self.middleCluster[left], left)


                    # initial update middle right
                    temp_middleCenterPoints, self.middleCluster[right] = self.initial_middle_update_right(temp_middleCenterPoints, self.middleCluster[right], right)



                elif self.middleFlag[left] != 0 and self.middleFlag[right] == 0:



                    # update middle left
                    temp_middleCenterPoints, self.middleCluster[left] = self.basic_middle_update_left(temp_middleCenterPoints, self.middleCluster[left], left)



                    # initial update middle right
                    temp_middleCenterPoints, self.middleCluster[right] = self.initial_middle_update_right(
                        temp_middleCenterPoints, self.middleCluster[right], right)
                elif self.middleFlag[left] == 0 and self.middleFlag[right] != 0:
                    # update middle right
                    temp_middleCenterPoints, self.middleCluster[right] = self.basic_middle_update_right(temp_middleCenterPoints, self.middleCluster[right], right)

                    # initial update middle left
                    temp_middleCenterPoints, self.middleCluster[left] = self.initial_middle_update_left(
                        temp_middleCenterPoints, self.middleCluster[left], left)


                elif self.middleFlag[left] != 0 and self.middleFlag[right] != 0:
                    # update middle left
                    temp_middleCenterPoints, self.middleCluster[left] = self.basic_middle_update_left(
                        temp_middleCenterPoints, self.middleCluster[left], left)

                    # update middle right
                    temp_middleCenterPoints, self.middleCluster[right] = self.basic_middle_update_right(
                        temp_middleCenterPoints, self.middleCluster[right], right)


        # Let's see if the clustering in the middle exists after such a long logical process
        if self.middleFlag[middle] != 0:
            # Check whether the current clustering meets the threshold boundary, and continue to correct if it does not
            self.check_boundary_middle()
