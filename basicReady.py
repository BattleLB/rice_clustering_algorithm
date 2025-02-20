
"""
    This file mainly contains some configuration before the main function is run
"""

from G_V import Global
from tools import Tool
from configure import Configure


class BasicReady:

    def __init__(self):
        self.GV = Global()
        self.TOOL = Tool()




    def main(self):

        CONFIGURE = Configure('power_shell')

        # Prints out the information for the current series of system configuration parameters
        CONFIGURE.printf()


