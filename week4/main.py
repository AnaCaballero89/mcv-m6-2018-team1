import sys

sys.path.append('../')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import *
from task1_1 import task1_1
from task1_2 import task1_2
from task2_1 import task2_1
from task2_2 import task2_2
from task2_3 import task2_3


def main():

    ###########
    # Task1.1 #
    ###########

    # Optical Flow with Block Matching
    task1_1()

    ###########
    # Task1.2 #
    ###########
    # Block Matching vs Other Techniques
    task1_2()

    ###########
    # Task2.1 #
    ###########

    # Video stabilization with Block Matching
    task2_1()

    ###########
    # Task2.2 #
    ###########

    # Block Matching Stabilization vs Other Techniques
    task2_2()

    ###########
    # Task2.3 #
    ###########

    # Stabilize your own video
    task2_3()


if __name__ == "__main__":
    main()