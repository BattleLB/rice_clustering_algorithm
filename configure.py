
"""
    This file contains some system variables and global configuration parameters
"""



class Configure():
    def __init__(self, terminal):
        self.TERMINAL_NAME = terminal
        self.ALGO_VER = 'ALGO_V180_'
        self.ALGORITHM_NAME = 'clustering_algorithm'
        self.DATASETS_NUMBER = '7042'

    # Prints out the information for the current series of system configuration parameters
    def printf(self):

        if self.TERMINAL_NAME == 'power_shell':
            print()
            print()
            print('######################################################################################################')
            print(f'#####The version number that is currently running and in use is----------{self.ALGO_VER}###################')
            print(f'#####The name of the algorithm that is currently running-----------------{self.ALGORITHM_NAME}###')
            print(f'#####The total number of images in the dataset to be detected in this test is as follows: {self.DATASETS_NUMBER}########')
            print('######################################################################################################')
            print()
            print()
            print("Let s start with the main function running part of the algorithm---------------")
        else:
            print()
            print('\033[31m########################################################################################\033[0m')
            print(f'\033[31m#####The version number that is currently running and in use is----------{self.ALGO_VER}#####\033[0m')






