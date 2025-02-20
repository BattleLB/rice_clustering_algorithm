"""
    This document mainly contains the implementation details of the algorithm classes and the related functions of the algorithm
"""

# Some of the capitalized letters in the document are explained below
# N is the number
# A is the all
# I is the information


import sys
import cv2
import tools
import G_V
import copy
import numpy as np


class Algorithm:
    def __init__(self, objf, savePath):
        self.detectionBoxesN = 0  # The number of detection boxes in the image currently being processed
        self.detectionBoxesA = objf  # Overall information for all detection frames
        self.detectionBoxesI = []  # Information about the detection frame in the image that is processed this time
        self.picNameLeft = ''  # The name of the image that is currently being processed
        self.integrationDetectionBoxes = []  # Integration information for the detection box
        self.integrationLines = []  # Integrated information for all the lines in this picture
        self.TOOL = tools.Tool()  # A toolkit for self-implementation
        self.savePath = savePath  # The path to the output of the image processing result
        self.linesN = 0  # The number of lines in this image
        self.GV = G_V.Global()

    # The operation of the main function of the algorithm
    def main(self, nowNum):
        while True:
            line = self.detectionBoxesA.readline()
            if not line:
                print('eof reached, exit...')
                break

            # Get the data of the detection frame of the first image that has been read so far
            nextNum = int(line[:8])
            if nextNum < nowNum:
                continue

            # If the number of detection frames in this image has been read, the calculation will begin
            if nextNum > nowNum:
                # Displays the current progress of data processing
                self.picNameLeft = './ytsplit/left' + format(nowNum, '06d') + '.jpg'
                print('now read pic ', self.picNameLeft)
                img = cv2.imread(self.picNameLeft)

                # Count the number of detection boxes in this image
                self.detectionBoxesN = len(self.detectionBoxesI)

                # First, the integration information for all the individual detection frames is calculated
                self.detectionBoxes_integrated_information_calculate()

                # After that, the integration information of all the lines is calculated
                self.lines_integration_information_calculated()

                # Three lines are selected from all the lines according to the range of slopes
                redLineSet_k = self.select_lines_k()

                # Three red lines are selected according to the angle range
                redLineSet_angle = self.select_lines_angle()

                # Three red lines are selected based on the range values of the endpoints
                redLineSet_endPoint = self.select_lines_endPoints()

                # A summary of the results of the red line selected for the three ways
                summaryResult = self.summary(redLineSet_k, redLineSet_angle, redLineSet_endPoint)

                # The most suitable three are selected from all the lines that have been summarized
                result = self.final_select(summaryResult)


                # Recording of data
                result, flag, Cnt = self.recording_data(result)

                # Draw the left, middle, and right lines
                img = self.TOOL.draw_left_middle_right(result, img, flag, Cnt)

                # Draw for both sides of the line disappearing in case of disappearance
                result, img = self.draw_disappearing_line(result, img)

                # Prediction for the final redline trace line
                img = self.predict(result, img)



                # The data of the redline searched is stored in a CSV file for subsequent comparison and viewing
                # for i in range(len(result)):
                #     summaryResult.append(self.picNameLeft)
                # self.TOOL.write_to_csv('./red_line_data', result)


                # Save the processed image
                # saveFn = self.savePath + 'left' + format(nowNum, '06d') + '.jpg'
                # # print('write to ', saveFn)
                # cv2.imwrite(saveFn, img)

                # Reset this series of data to zero
                self.reset()
                nowNum = nextNum

            # Add vertex coordinates for the detection box
            self.detectionBoxesI.append([int(line[8:12]), int(line[12:16]), int(line[16:20]), int(line[20:24])])

        self.detectionBoxesA.close()

    # Reset this series of data to zero
    def reset(self):
        self.detectionBoxesI = []
        self.integrationDetectionBoxes = []
        self.integrationLines = []

    # Calculation of the integrated information of the detection box
    def detectionBoxes_integrated_information_calculate(self):
        tempList = []
        # Let's start with a description
        self.integrationDetectionBoxes.append(['The integrated information of the detection frame is arranged in '
                                               'order: the coordinates of the upper left vertex, the lower right '
                                               'vertex and the center point of the detection frame'])
        for i in range(self.detectionBoxesN):
            tempList.append(['detection box  ' + str(i + 1)])
            tempList.append(self.detectionBoxesI[i][0:2])
            tempList.append(self.detectionBoxesI[i][2:4])
            # Calculates the coordinates of the center point of the detection box
            tempList.append(self.TOOL.calculate_center_points(tempList[1], tempList[2]))
            self.integrationDetectionBoxes.append(tempList)
            tempList = []

    # After that, the integration information of all the lines is calculated
    def lines_integration_information_calculated(self):
        self.integrationLines.append(['The following is the integration information of all the lines in the processed '
                                      'picture, in order: the integration information of the detection box 1, '
                                      'the integration information of the detection box 2, the slope and b value of '
                                      'the line, the angle formed by the line and the lower baseline, '
                                      'the coordinates of the two endpoints of the line, and The sum of the '
                                      'coordinates of the center point of the two detection frames'])

        for i in range(self.detectionBoxesN):
            for j in range(i + 1, self.detectionBoxesN):
                tempList1 = self.integrationDetectionBoxes[i + 1]
                tempList2 = self.integrationDetectionBoxes[j + 1]
                # Calculate the slope of the line with the b-value
                KB = self.TOOL.get_k_b(tempList1[3], tempList2[3])

                # Once you have calculated the slope and b value of the line,
                # let's start calculating the angle and endpoint values
                if KB is None:
                    angle = None
                    endPoints = None
                    differenceValue = None
                else:
                    # calculate angle
                    angle = self.TOOL.calculate_angle(KB)
                    # calculate endPoints
                    endPoints = self.TOOL.calculate_endpoints_line(KB[0], KB[1])
                    # calculate difference value
                    differenceValue = self.TOOL.difference_endpoints(endPoints)

                # All calculated information is stored
                self.integrationLines.append([tempList1, tempList2, KB, angle, endPoints, differenceValue])

        self.linesN = len(self.integrationLines)

    # Three lines are selected from all the lines according to the range of slopes
    def select_lines_k(self):

        result = []

        '''red middle'''
        pendingCollection = []
        # First, select a line with a slope range between -1 and 1
        for i in range(1, self.linesN):
            if self.integrationLines[i][2] is not None:
                if self.GV.kLowerLimit <= self.integrationLines[i][2][self.GV.kIndex] <= self.GV.kUpperLimit:
                    pendingCollection.append(self.integrationLines[i])

        # Calculate the difference value
        differenceValue = []
        for i in range(len(pendingCollection)):
            sumValue = abs(pendingCollection[i][0][3][self.GV.xIndex] - self.GV.CENTER) + abs(
                pendingCollection[i][1][3][self.GV.xIndex] - self.GV.CENTER)
            differenceValue.append(sumValue)

        # The minimum value is selected and its subscript in the original array is returned
        min_index = self.TOOL.get_min_value(differenceValue)

        # When you are selected for a line that meets the criteria, add it to the collection
        if min_index != -1:
            temp_select = pendingCollection[min_index]
            if temp_select[5] < self.GV.deviationLimits:
                result.append(temp_select)

        temp_integrationLines = copy.deepcopy(self.integrationLines)
        # The next step is to remove the line that is related to its detection box from the collection
        if result:
            detectionBox1 = result[0][0]
            detectionBox2 = result[0][1]
            # Find out where members need to be removed
            index = np.zeros(len(temp_integrationLines))
            for i in range(1, self.linesN):
                if temp_integrationLines[i][0][0] == detectionBox1[0] or temp_integrationLines[i][1][0] == \
                        detectionBox1[0]:
                    index[i] = 1
                if temp_integrationLines[i][0][0] == detectionBox2[0] or temp_integrationLines[i][1][0] == \
                        detectionBox2[0]:
                    index[i] = 1
            count = 0
            for i in range(len(index)):
                if index[i] == 1:
                    del temp_integrationLines[i - count]
                    count += 1

        ''' red left'''
        # Start selecting the second line
        pendingCollection = []
        # select a line with a slope range between -1 and 1
        for i in range(1, len(temp_integrationLines)):
            if temp_integrationLines[i][2] is not None:
                if self.GV.kLowerLimit <= temp_integrationLines[i][2][self.GV.kIndex] <= self.GV.kUpperLimit:
                    pendingCollection.append(temp_integrationLines[i])

        if pendingCollection:
            # When the slope is consistent, another round of filtering is performed according to the coordinates of
            # the end of the line When you select the left line, you need the upper endpoint to get an abscissa that
            # is greater than the abscissa of the lower endpoint

            temp = []
            for i in range(len(pendingCollection)):
                if pendingCollection[i][4][0][self.GV.xIndex] >= pendingCollection[i][4][1][self.GV.xIndex] and (
                        pendingCollection[i][4][0][self.GV.xIndex] <= (self.GV.CENTER + self.GV.trimOffset)):
                    temp.append(pendingCollection[i])
            # Finally, the line with the smallest absolute value from the center pixel is selected from the remaining
            # lines

            value = []
            for i in range(len(temp)):
                value.append(abs(temp[i][4][0][self.GV.xIndex] - (self.GV.CENTER + self.GV.trimOffset)))

            min_index = self.TOOL.get_min_value(value)

            if min_index != -1:
                result.append(temp[min_index])
                # delete
                detectionBox1 = temp[min_index][0]
                detectionBox2 = temp[min_index][1]
                # Find out where members need to be removed
                index = np.zeros(len(temp_integrationLines))
                for i in range(1, len(temp_integrationLines)):
                    if temp_integrationLines[i][0][0] == detectionBox1[0] or temp_integrationLines[i][1][0] == \
                            detectionBox1[0]:
                        index[i] = 1
                    if temp_integrationLines[i][0][0] == detectionBox2[0] or temp_integrationLines[i][1][0] == \
                            detectionBox2[0]:
                        index[i] = 1
                count = 0
                for i in range(len(index)):
                    if index[i] == 1:
                        del temp_integrationLines[i - count]
                        count += 1

        ''' red right'''
        # Start selecting the third line
        pendingCollection = []
        # select a line with a slope range between -1 and 1
        for i in range(1, len(temp_integrationLines)):
            if temp_integrationLines[i][2] is not None:
                if self.GV.kLowerLimit <= temp_integrationLines[i][2][self.GV.kIndex] <= self.GV.kUpperLimit:
                    pendingCollection.append(temp_integrationLines[i])

        if pendingCollection:

            temp = []
            for i in range(len(pendingCollection)):
                if pendingCollection[i][4][0][self.GV.xIndex] <= pendingCollection[i][4][1][self.GV.xIndex] and (
                        pendingCollection[i][4][0][self.GV.xIndex] >= (self.GV.CENTER + self.GV.trimOffset)):
                    temp.append(pendingCollection[i])

            value = []
            for i in range(len(temp)):
                value.append(abs(temp[i][4][0][self.GV.xIndex] - (self.GV.CENTER + self.GV.trimOffset)))

            min_index = self.TOOL.get_min_value(value)
            if min_index != -1:
                result.append(temp[min_index])

        return result

    # Three red lines are selected according to the angle range
    def select_lines_angle(self):

        result = []

        ''' red middle '''
        # Select a line in the middle according to the datum of the angle
        pendingCollection = []
        for i in range(1, self.linesN):
            if self.integrationLines[i][3] is not None:
                if (self.GV.angleDatum - self.GV.angleFSize) <= self.integrationLines[i][3] <= (
                        self.GV.angleDatum + self.GV.angleFSize):
                    pendingCollection.append(self.integrationLines[i])

        # Calculate the difference value
        differenceValue = []
        for i in range(len(pendingCollection)):
            sumValue = abs(pendingCollection[i][0][3][self.GV.xIndex] - self.GV.CENTER) + abs(
                pendingCollection[i][1][3][self.GV.xIndex] - self.GV.CENTER)
            differenceValue.append(sumValue)

        # The minimum value is selected and its subscript in the original array is returned
        min_index = self.TOOL.get_min_value(differenceValue)

        # When you are selected for a line that meets the criteria, add it to the collection
        if min_index != -1:
            temp_select = pendingCollection[min_index]
            if temp_select[5] < self.GV.deviationLimits:
                result.append(temp_select)

        temp_integrationLines = copy.deepcopy(self.integrationLines)
        # The next step is to remove the line that is related to its detection box from the collection
        if result:
            detectionBox1 = result[0][0]
            detectionBox2 = result[0][1]
            # Find out where members need to be removed
            index = np.zeros(len(temp_integrationLines))
            for i in range(1, self.linesN):
                if temp_integrationLines[i][0][0] == detectionBox1[0] or temp_integrationLines[i][1][0] == \
                        detectionBox1[0]:
                    index[i] = 1
                if temp_integrationLines[i][0][0] == detectionBox2[0] or temp_integrationLines[i][1][0] == \
                        detectionBox2[0]:
                    index[i] = 1
            count = 0
            for i in range(len(index)):
                if index[i] == 1:
                    del temp_integrationLines[i - count]
                    count += 1

        ''' red left '''
        # Start selecting the second line
        pendingCollection = []
        # select a line with a slope range between -1 and 1
        for i in range(1, len(temp_integrationLines)):
            if temp_integrationLines[i][3] is not None:
                if temp_integrationLines[i][3] >= (self.GV.angleDatum + self.GV.angleFSize):
                    pendingCollection.append(temp_integrationLines[i])

        if pendingCollection:

            temp = []
            for i in range(len(pendingCollection)):
                if pendingCollection[i][4][0][self.GV.xIndex] >= pendingCollection[i][4][1][self.GV.xIndex] and (
                        pendingCollection[i][4][0][self.GV.xIndex] <= (self.GV.CENTER + self.GV.trimOffset)):
                    temp.append(pendingCollection[i])

            value = []
            for i in range(len(temp)):
                value.append(abs(temp[i][4][0][self.GV.xIndex] - (self.GV.CENTER + self.GV.trimOffset)))

            min_index = self.TOOL.get_min_value(value)

            if min_index != -1:
                result.append(temp[min_index])
                # delete
                detectionBox1 = temp[min_index][0]
                detectionBox2 = temp[min_index][1]
                # Find out where members need to be removed
                index = np.zeros(len(temp_integrationLines))
                for i in range(1, len(temp_integrationLines)):
                    if temp_integrationLines[i][0][0] == detectionBox1[0] or temp_integrationLines[i][1][0] == \
                            detectionBox1[0]:
                        index[i] = 1
                    if temp_integrationLines[i][0][0] == detectionBox2[0] or temp_integrationLines[i][1][0] == \
                            detectionBox2[0]:
                        index[i] = 1
                count = 0
                for i in range(len(index)):
                    if index[i] == 1:
                        del temp_integrationLines[i - count]
                        count += 1

        ''' red right  '''
        # Start selecting the third line
        pendingCollection = []
        # select a line with a slope range between -1 and 1
        for i in range(1, len(temp_integrationLines)):
            if temp_integrationLines[i][3] is not None:
                if temp_integrationLines[i][3] <= (self.GV.angleDatum - self.GV.angleFSize):
                    pendingCollection.append(temp_integrationLines[i])

        if pendingCollection:

            temp = []
            for i in range(len(pendingCollection)):
                if pendingCollection[i][4][0][self.GV.xIndex] <= pendingCollection[i][4][1][self.GV.xIndex] and (
                        pendingCollection[i][4][0][self.GV.xIndex] >= (self.GV.CENTER + self.GV.trimOffset)):
                    temp.append(pendingCollection[i])

            value = []
            for i in range(len(temp)):
                value.append(abs(temp[i][4][0][self.GV.xIndex] - (self.GV.CENTER + self.GV.trimOffset)))

            min_index = self.TOOL.get_min_value(value)
            if min_index != -1:
                result.append(temp[min_index])

        return result

    # Three red lines are selected based on the range values of the endpoints
    def select_lines_endPoints(self):

        result = []

        ''' red middle'''
        # Pick the red line based on the endpoint value
        pendingCollection = []
        for i in range(1, self.linesN):
            if self.integrationLines[i][4] is not None:
                if (self.GV.CENTER + self.GV.trimOffset - self.GV.endPointFSize) <= self.integrationLines[i][4][0][
                    self.GV.xIndex] <= (
                        self.GV.CENTER + self.GV.trimOffset + self.GV.endPointFSize):
                    pendingCollection.append(self.integrationLines[i])

        # Calculate the difference value
        differenceValue = []
        for i in range(len(pendingCollection)):
            sumValue = abs(pendingCollection[i][0][3][self.GV.xIndex] - self.GV.CENTER) + abs(
                pendingCollection[i][1][3][self.GV.xIndex] - self.GV.CENTER)
            differenceValue.append(sumValue)

        # The minimum value is selected and its subscript in the original array is returned
        min_index = self.TOOL.get_min_value(differenceValue)
        if min_index != -1:
            temp_select = pendingCollection[min_index]
            if temp_select[5] < self.GV.deviationLimits:
                result.append(temp_select)

        temp_integrationLines = copy.deepcopy(self.integrationLines)
        # The next step is to remove the line that is related to its detection box from the collection
        if result:
            detectionBox1 = result[0][0]
            detectionBox2 = result[0][1]
            # Find out where members need to be removed
            index = np.zeros(len(temp_integrationLines))
            for i in range(1, self.linesN):
                if temp_integrationLines[i][0][0] == detectionBox1[0] or temp_integrationLines[i][1][0] == \
                        detectionBox1[0]:
                    index[i] = 1
                if temp_integrationLines[i][0][0] == detectionBox2[0] or temp_integrationLines[i][1][0] == \
                        detectionBox2[0]:
                    index[i] = 1
            count = 0
            for i in range(len(index)):
                if index[i] == 1:
                    del temp_integrationLines[i - count]
                    count += 1

        ''' red left'''
        # Start selecting the second line
        pendingCollection = []
        # select a line with a slope range between -1 and 1
        for i in range(1, len(temp_integrationLines)):
            if temp_integrationLines[i][4] is not None:
                if temp_integrationLines[i][4][0][self.GV.xIndex] <= (
                        self.GV.CENTER + self.GV.trimOffset + self.GV.endPointFSize) and temp_integrationLines[i][4][1][self.GV.xIndex] <= (self.GV.CENTER + self.GV.trimOffset + self.GV.endPointFSize):
                    pendingCollection.append(temp_integrationLines[i])

        if pendingCollection:
            value = []
            for i in range(len(pendingCollection)):
                value.append(
                    abs(pendingCollection[i][4][0][self.GV.xIndex] - (self.GV.CENTER + self.GV.trimOffset)) + abs(
                        pendingCollection[i][4][1][self.GV.xIndex] - (self.GV.CENTER + self.GV.trimOffset)))

            min_index = self.TOOL.get_min_value(value)

            if min_index != -1:
                result.append(pendingCollection[min_index])
                # delete
                detectionBox1 = pendingCollection[min_index][0]
                detectionBox2 = pendingCollection[min_index][1]
                # Find out where members need to be removed
                index = np.zeros(len(temp_integrationLines))
                for i in range(1, len(temp_integrationLines)):
                    if temp_integrationLines[i][0][0] == detectionBox1[0] or temp_integrationLines[i][1][0] == \
                            detectionBox1[0]:
                        index[i] = 1
                    if temp_integrationLines[i][0][0] == detectionBox2[0] or temp_integrationLines[i][1][0] == \
                            detectionBox2[0]:
                        index[i] = 1
                count = 0
                for i in range(len(index)):
                    if index[i] == 1:
                        del temp_integrationLines[i - count]
                        count += 1

        ''' red right '''
        # Start selecting the third line
        pendingCollection = []
        # select a line with a slope range between -1 and 1
        for i in range(1, len(temp_integrationLines)):
            if temp_integrationLines[i][4] is not None:
                if temp_integrationLines[i][4][0][self.GV.xIndex] >= (
                        self.GV.CENTER + self.GV.trimOffset - self.GV.endPointFSize) and temp_integrationLines[i][4][1][
                    self.GV.xIndex] >= (self.GV.CENTER + self.GV.trimOffset - self.GV.endPointFSize):
                    pendingCollection.append(temp_integrationLines[i])

        if pendingCollection:
            value = []
            for i in range(len(pendingCollection)):
                value.append(
                    abs(pendingCollection[i][4][0][self.GV.xIndex] - (self.GV.CENTER + self.GV.trimOffset)) + abs(
                        pendingCollection[i][4][1][self.GV.xIndex] - (self.GV.CENTER + self.GV.trimOffset)))

            min_index = self.TOOL.get_min_value(value)
            if min_index != -1:
                result.append(pendingCollection[min_index])

        return result

    # A summary of the results of the red line selected for the three ways
    def summary(self, kLine, angleLine, endPointLine):

        result = []

        for i in range(len(kLine)):
            result.append(kLine[i])
        for i in range(len(angleLine)):
            result.append(angleLine[i])
        for i in range(len(endPointLine)):
            result.append(endPointLine[i])

        return result



    # The most suitable three are selected from all the lines that have been summarized
    def final_select(self, summaryResult):

        result = []

        ''' red left '''
        # First of all, the data information of the duplicate lines is removed
        index = np.zeros(len(summaryResult))
        for i in range(len(summaryResult)):
            detectionBox1 = summaryResult[i][0]
            detectionBox2 = summaryResult[i][1]
            for j in range(i+1, len(summaryResult)):
                if summaryResult[j][0][0] == detectionBox1[0] and summaryResult[j][1][0] == detectionBox2[0]:
                    index[j] = 1
        count = 0
        for i in range(len(index)):
            if index[i] == 1:
                del summaryResult[i - count]
                count += 1

        tempSummaryResult = copy.deepcopy(summaryResult)
        # Divide the processed summary line into the left part
        leftPart = []
        for i in range(len(summaryResult)):
            if (summaryResult[i][4][0][self.GV.xIndex] <= (self.GV.CENTER + self.GV.trimOffset + self.GV.endPointFSize)
                    and summaryResult[i][4][1][self.GV.xIndex] <= (self.GV.CENTER + self.GV.trimOffset + self.GV.endPointFSize))\
                    and summaryResult[i][2][self.GV.kIndex] <= 0:
                leftPart.append(summaryResult[i])

        if leftPart:
            # Take the line with the smallest slope value as the red line on the left
            value = []
            for i in range(len(leftPart)):
                value.append(leftPart[i][2][self.GV.kIndex])

            min_index = self.TOOL.get_min_value(value)

            if min_index != -1:
                result.append(leftPart[min_index])
                # Deletes the selected line from the original summary set
                detectionBox1 = leftPart[min_index][0]
                detectionBox2 = leftPart[min_index][1]
                for i in range(len(tempSummaryResult)):
                    if detectionBox1[0] == tempSummaryResult[i][0][0] and detectionBox2[0] == tempSummaryResult[i][1][0]:
                        del tempSummaryResult[i]
                        break

        ''' red right '''
        # Divide the processed summary line into the right part
        rightPart = []
        for i in range(len(tempSummaryResult)):
            if (tempSummaryResult[i][4][0][self.GV.xIndex] >= (self.GV.CENTER + self.GV.trimOffset - self.GV.endPointFSize)
                    and tempSummaryResult[i][4][1][self.GV.xIndex] >= (self.GV.CENTER + self.GV.trimOffset - self.GV.endPointFSize))\
                    and tempSummaryResult[i][2][self.GV.kIndex] >= 0:
                rightPart.append(tempSummaryResult[i])


        if rightPart:
            # Take the line with the smallest slope value as the red line on the left
            value = []
            for i in range(len(rightPart)):
                value.append(rightPart[i][2][self.GV.kIndex])

            # Choose the maximum value
            max_index = self.TOOL.get_max_value(value)
            if max_index != -1:
                result.append(rightPart[max_index])
                # Deletes the selected line from the original summary set
                detectionBox1 = rightPart[max_index][0]
                detectionBox2 = rightPart[max_index][1]
                for i in range(len(tempSummaryResult)):
                    if detectionBox1[0] == tempSummaryResult[i][0][0] and detectionBox2[0] == tempSummaryResult[i][1][0]:
                        del tempSummaryResult[i]
                        break

        ''' red middle'''

        # Excludes lines from previous collections based on intersection relationships and intersection coordinates
        if tempSummaryResult and result != []:
            index = np.zeros(len(tempSummaryResult))
            for i in range(len(tempSummaryResult)):
                for j in range(len(result)):
                    flag, intersectionPoints = self.TOOL.line_segment_intersection(tempSummaryResult[i][4], result[j][4])
                    if flag and self.GV.intersectionLowLimit <= intersectionPoints[self.GV.yIndex] <= self.GV.intersectionUpperLimit:
                        index[i] = 1

            count = 0
            for i in range(len(index)):
                if index[i] == 1:
                    del tempSummaryResult[i - count]
                    count += 1

            # Select the line with the smallest distance as the last line
            if tempSummaryResult:
                value = []
                for i in range(len(tempSummaryResult)):
                    value1 = self.TOOL.calculate_distance(tempSummaryResult[i][4][0], self.GV.centerPoints)
                    value2 = self.TOOL.calculate_distance(tempSummaryResult[i][4][1], self.GV.centerPoints)
                    value.append(value1 + value2)

                min_index = self.TOOL.get_min_value(value)
                if min_index != -1:
                    result.append(tempSummaryResult[min_index])

        # For this result, the left, center, and right order is performed
        result = self.left_center_right_sort(result)

        return result



    # For this result, the left, center, and right order is performed
    def left_center_right_sort(self, result):

        centerPoints = []
        for i in range(len(result)):
            centerPoints.append(self.TOOL.calculate_center_points(result[i][4][0], result[i][4][1]))
        tempResult = []

        if len(centerPoints) < 3:
            flag = 0
            for i in range(len(centerPoints)):
                if centerPoints[i][self.GV.xIndex] < self.GV.leftBase:
                    result[i].append(1)
                    tempResult.append(result[i])
                    flag = 1
            if flag == 0:
                tempResult.append(self.GV.invalidData)
                flag = 1


            for i in range(len(centerPoints)):
                if self.GV.leftBase <= centerPoints[i][self.GV.xIndex] <= self.GV.rightBase:
                    result[i].append(2)
                    tempResult.append(result[i])
                    flag = 2
            if flag == 1:
                tempResult.append(self.GV.invalidData)
                flag = 2

            for i in range(len(centerPoints)):
                if centerPoints[i][self.GV.xIndex] > self.GV.rightBase:
                    result[i].append(3)
                    tempResult.append(result[i])
                    flag = 3
            if flag == 2:
                tempResult.append(self.GV.invalidData)
        else:

            for i in range(len(centerPoints)):
                if centerPoints[i][self.GV.xIndex] < self.GV.leftBase:
                    result[i].append(1)
                    tempResult.append(result[i])

            for i in range(len(centerPoints)):
                if self.GV.leftBase <= centerPoints[i][self.GV.xIndex] <= self.GV.rightBase:
                    result[i].append(2)
                    tempResult.append(result[i])

            for i in range(len(centerPoints)):
                if centerPoints[i][self.GV.xIndex] > self.GV.rightBase:
                    result[i].append(3)
                    tempResult.append(result[i])

        # Again, adjust the data to make it easier to draw multiple tracking red lines
        count = 0
        for i in range(len(tempResult)):
            if tempResult[i] != self.GV.invalidData:
                if tempResult[i][len(tempResult[i]) - 1] == 2:
                    count += 1

        if count >= 2:
            # Select the line with a smaller slope value as the trace line, and the other lines as the line to the left or right of it
            value = []
            for i in range(len(tempResult)):
                if tempResult[i][len(tempResult[i]) - 1] == 2:
                    value.append(abs(tempResult[i][2][self.GV.kIndex] - 0))
                else:
                    value.append(999999)

            min_index = self.TOOL.get_min_value(value)
            center = self.TOOL.calculate_center_points(tempResult[min_index][4][0], tempResult[min_index][4][1])
            for i in range(len(tempResult)):
                if tempResult[i] != self.GV.invalidData:
                    if tempResult[i][len(tempResult[i]) - 1] == 2:
                        otherCenter = self.TOOL.calculate_center_points(tempResult[i][4][0], tempResult[i][4][1])
                        if otherCenter[0] <= center[0]:
                            tempResult[i][len(tempResult[i]) - 1] = 1
                        elif otherCenter[0] >= center[0]:
                            tempResult[i][len(tempResult[i]) - 1] = 3



        return tempResult


    # Recording of data
    def recording_data(self, lineSet):
        flag = False

        # First of all, count how many lines there are on the left, middle, and right
        count = [0, 0, 0]
        for i in range(len(lineSet)):
            if lineSet[i] != self.GV.invalidData:
                if lineSet[i][len(lineSet[i]) - 1] == 1:
                    count[0] += 1
                elif lineSet[i][len(lineSet[i]) - 1] == 2:
                    count[1] += 1
                elif lineSet[i][len(lineSet[i]) - 1] == 3:
                    count[2] += 1

        # For the left and right, if there is an excess, the optimal data is retained
        if count[0] >= 2:

            value = []
            for i in range(len(lineSet)):
                if lineSet[i][len(lineSet[i]) - 1] == 1:
                    value.append(abs(lineSet[i][2][self.GV.kIndex] - 0))
                else:
                    value.append(-9999999)
            max_index = self.TOOL.get_max_value(value)
            for i in range(len(lineSet)):
                if lineSet[i][len(lineSet[i]) - 1] == 1:
                    if i != max_index:
                        lineSet[i] = self.GV.invalidData

        if count[2] >= 2:

            value = []
            for i in range(len(lineSet)):
                if lineSet[i][len(lineSet[i]) - 1] == 3:
                    value.append(abs(lineSet[i][2][self.GV.kIndex] - 0))
                else:
                    value.append(-9999999)

            max_index = self.TOOL.get_max_value(value)
            for i in range(len(lineSet)):
                if lineSet[i][len(lineSet[i]) - 1] == 3:
                    if i != max_index:
                        lineSet[i] = self.GV.invalidData



        # Add the organized information to the history
        for line in range(len(self.GV.recordLine)):
            for i in range(len(self.GV.recordLine[line]) - 1):
                self.GV.recordLine[line][i] = self.GV.recordLine[line][i + 1]
            self.GV.recordLine[line][len(self.GV.recordLine[line]) - 1] = self.GV.invalidData


        for i in range(len(lineSet)):
            if lineSet[i][len(lineSet[i]) - 1] == 1:
                self.GV.recordLine[0][len(self.GV.recordLine[0]) - 1] = lineSet[i]
            elif lineSet[i][len(lineSet[i]) - 1] == 2:
                self.GV.recordLine[1][len(self.GV.recordLine[1]) - 1] = lineSet[i]
            elif lineSet[i][len(lineSet[i]) - 1] == 3:
                self.GV.recordLine[2][len(self.GV.recordLine[2]) - 1] = lineSet[i]

        Cnt = 1
        # For data that did not find intermediate trace lines in this search, look back and see if there is data for
        # the previous trace line available (without exceeding the number limit).
        if count[1] == 0:

            data = []
            for i in range(self.GV.trackCnt):
                data.append(self.GV.recordLine[1][len(self.GV.recordLine[1]) - 1 - i])
            value = []
            for i in range(len(data)):
                if data[i] != self.GV.invalidData and data[i] != [0.0, 0.0]:
                    value.append(abs(data[i][2][self.GV.kIndex] - 0))
                else:
                    value.append(99999)
            min_index = self.TOOL.get_min_value(value)


            for i in range(len(value)):
                if value[i] == 99999:
                    Cnt += 1

            if Cnt != 5:
                for i in range(len(lineSet)):
                    if lineSet[i] == self.GV.invalidData:
                        lineSet[i] = data[min_index]
                        flag = True
                        break

        return lineSet, flag, Cnt

    # Draw for both sides of the line disappearing in case of disappearance
    def draw_disappearing_line(self, lineSet, img):

        # Let's take a look at whether the lines on both sides of the search are legitimate, and if they are not, they will be discarded
        for i in range(len(lineSet)):
            if lineSet[i] != self.GV.invalidData:
                if lineSet[i][len(lineSet[i]) - 1] == 1:
                    if lineSet[i][2][self.GV.kIndex] > 0:
                        lineSet[i] = self.GV.invalidData

        for i in range(len(lineSet)):
            if lineSet[i] != self.GV.invalidData:
                if lineSet[i][len(lineSet[i]) - 1] == 3:
                    if lineSet[i][2][self.GV.kIndex] < 0:
                        lineSet[i] = self.GV.invalidData

        # Count whether there are still lines on the left and right sides at this time
        count = [0, 0, 0]
        for i in range(len(lineSet)):
            if lineSet[i] != self.GV.invalidData:
                if lineSet[i][len(lineSet[i]) - 1] == 1:
                    count[0] += 1
                elif lineSet[i][len(lineSet[i]) - 1] == 2:
                    count[1] += 1
                elif lineSet[i][len(lineSet[i]) - 1] == 3:
                    count[2] += 1

        if count[0] == 0:
            data = []
            for i in range(self.GV.disappearCnt):
                data.append(self.GV.recordLine[0][len(self.GV.recordLine[0]) - 2 - i])
            value = []
            for i in range(len(data)):
                if data[i] != self.GV.invalidData and data[i] != [0.0, 0.0] and data[i][2][self.GV.kIndex] < 0:
                    value.append(abs(data[i][2][self.GV.kIndex] - 0))
                else:
                    value.append(-99999)
            max_index = self.TOOL.get_max_value(value)

            # Count how many times it has disappeared this time
            Cnt = 1
            for i in range(len(value)):
                if value[i] == -99999:
                    Cnt += 1
                else:
                    break
            if Cnt != 5:
                img = self.TOOL.cnt_dotted_line(data[max_index][4][0], data[max_index][4][1], img, Cnt)
                # This dashed line is added to the collection to facilitate subsequent prediction operations for intermediate trace lines
                for i in range(len(lineSet)):
                    if lineSet[i] == self.GV.invalidData:
                        lineSet[i] = data[max_index]
                        break


        if count[2] == 0:
            data = []
            for i in range(self.GV.disappearCnt):
                data.append(self.GV.recordLine[2][len(self.GV.recordLine[2]) - 2 - i])
            value = []
            for i in range(len(data)):
                if data[i] != self.GV.invalidData and data[i] != [0.0, 0.0] and data[i][2][self.GV.kIndex] > 0:
                    value.append(abs(data[i][2][self.GV.kIndex] - 0))
                else:
                    value.append(-99999)
            max_index = self.TOOL.get_max_value(value)

            Cnt = 1
            # Count how many times it has disappeared this time
            for i in range(len(value)):
                if value[i] == -99999:
                    Cnt += 1
                else:
                    break

            if Cnt != 5:
                img = self.TOOL.cnt_dotted_line(data[max_index][4][0], data[max_index][4][1], img, Cnt)
                # This dashed line is added to the collection to facilitate subsequent prediction operations for intermediate trace lines
                for i in range(len(lineSet)):
                    if lineSet[i] == self.GV.invalidData:
                        lineSet[i] = data[max_index]
                        break

        return lineSet, img

    # Prediction for the final redline trace line
    def predict(self, lineSet, img):


        # First of all, let's see if the tracking line exists, and if it exists, there is no need to predict it
        count = [0, 0, 0]
        for i in range(len(lineSet)):
            if lineSet[i] != self.GV.invalidData:
                if lineSet[i][len(lineSet[i]) - 1] == 1:
                    count[0] += 1
                elif lineSet[i][len(lineSet[i]) - 1] == 2:
                    count[1] += 1
                elif lineSet[i][len(lineSet[i]) - 1] == 3:
                    count[2] += 1

        if count[1] == 0 and count[0] == 1 and count[2] == 1:
            data_left = []
            data_right = []
            # Take out the line on the left and the line on the right
            for i in range(len(lineSet)):
                if lineSet[i] != self.GV.invalidData:
                    if lineSet[i][len(lineSet[i]) - 1] == 1:
                        data_left = lineSet[i][4]
                    elif lineSet[i][len(lineSet[i]) - 1] == 3:
                        data_right = lineSet[i][4]
            predict_points = [[int((data_left[0][self.GV.xIndex] + data_right[0][self.GV.kIndex]) / 2), 0], [int((data_left[1][self.GV.xIndex] + data_right[1][self.GV.xIndex]) / 2), 479]]
            img = self.TOOL.draw_blue(predict_points[0][self.GV.xIndex], predict_points[0][self.GV.yIndex], predict_points[1][self.GV.xIndex], predict_points[1][self.GV.yIndex], img)
            return img
        else:
            return img