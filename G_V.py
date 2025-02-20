"""
    The file contains some commonly used global variables
"""


class Global:
    def __init__(self):
        self.xIndex = 0
        self.yIndex = 1

        self.layerUpper = 94  # Hierarchical baseline
        self.layerMiddle = 154
        self.layerLower = 214
        self.layerFSize = 5  # The value that fluctuates up and down when the hierarchical data is calculated


        self.leftIndex = 0
        self.middleIndex = 1
        self.rightIndex = 2

        self.lowInitMaxSize = 80
        self.lowMaxSize = 90
        self.middleInitMaxSize = 60
        self.middleMaxSize = 70

        self.oldWeights = 0.4
        self.newWeights = 0.6


        self.limitLowLeft = 10
        self.limitLowRight = 10
        self.limitMiddleLeft = 5
        self.limitMiddleRight = 5

        self.updateLimitLowLeft = 100
        self.updateLimitLowRight = 100
        self.updateLimitMiddleLeft = 80
        self.updateLimitMiddleRight = 80


        self.trimOffset = 100  # Image abscissa offset



        self.yellow = (0, 255, 255)
        self.blue = (255, 0, 0)
        self.brown = (42, 42, 165)

        self.longEdgeLength = 60
        self.shortEdgeLength = 20
        self.scaleFactor = 1

        self.deviation = 2

