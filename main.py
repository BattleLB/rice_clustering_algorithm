
"""
    This file mainly contains the main operations and main logic architecture of the algorithm program runtime
"""


from basicReady import BasicReady
from clusteringAlgorithm_2 import CLUAlgorithm2





if __name__ == '__main__':

    # Do some preparation before the algorithm runs
    READY = BasicReady()
    READY.main()

    # Create the most important algorithm class
    ALGORITHM = CLUAlgorithm2('objlist.txt')

    # The operation of the main function of the algorithm
    ALGORITHM.main()







