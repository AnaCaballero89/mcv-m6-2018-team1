import sys
sys.path.append('../week1/')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from opticalFlowMetrics import opticalFlowMetrics
from utils import *
from task1 import task1
from task2 import task2
from task3 import task3
from task4 import task4


#########
# Task1 #
#########

# Gaussian distribution + evaluation
task1()


#########
# Task2 #
#########

# Recursive Gaussian modeling + Evaluate and comparison to non-recursive
task2()


#########
# Task3 #
#########

# Comparison with state-of-the-art
task3()


#########
# Task4 #
#########

# Color sequences
task4()
