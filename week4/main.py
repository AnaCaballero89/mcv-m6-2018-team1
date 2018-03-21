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
    dataset = 'traffic'

    tr_frmStart = None
    tr_frmEnd = None
    te_frmStart = None
    te_frmEnd = None


    if dataset == 'traffic':
        tr_frmStart = 950
        tr_frmEnd = 1000
        te_frmStart = 1001
        te_frmEnd = 1050
        MOGthreshold = 330
        inputpath = '../datasets/' + dataset + '/input/'
        groundTruthPath = '../datasets/' + dataset + '/groundtruth/'
    elif dataset == 'seq_045':
        inputpath = '../datasets/optical_flow_w1task3/' + dataset + '/input/'
        groundTruthPath = '../datasets/optical_flow_w1task3/' + dataset + '/groundtruth/'
    elif dataset == 'seq_157':
        inputpath = '../datasets/optical_flow_w1task3/' + dataset + '/input/'
        groundTruthPath = '../datasets/optical_flow_w1task3/' + dataset + '/groundtruth/'
    else:
        print "You haven't defined the right dataset. Options are: highway, fall or traffic."
        exit(0)

    ###########
    # Task1.1 #
    ###########

    # Optical Flow with Block Matching
    #task1_1(inputpath, groundTruthPath, dataset)

    ###########
    # Task1.2 #
    ###########
    # Block Matching vs Other Techniques
    task1_2()

    ###########
    # Task2.1 #
    ###########

    # Video stabilization with Block Matching
    task2_1(inputpath,groundTruthPath, dataset, tr_frmStart=tr_frmStart, tr_frmEnd=tr_frmEnd, te_frmStart=te_frmStart, te_frmEnd=te_frmEnd)

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