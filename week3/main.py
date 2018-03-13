import sys
sys.path.append('../')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import *
from task1 import task1
from task2 import task2
from task3 import task3
from task4 import task4
from task5 import task5


dataset = 'traffic'  # 'fall', 'highway', 'traffic'

inputpath='../datasets/' + dataset + '/input/'
groundTruthPath='../datasets/' + dataset + '/groundtruth/'

if dataset == 'highway':
    tr_frmStart = 1050
    tr_frmEnd = 1200
    te_frmStart = 1201
    te_frmEnd = 1350
    MOGthreshold = 10

elif dataset == 'fall':
    tr_frmStart = 1460
    tr_frmEnd = 1510
    te_frmStart = 1511
    te_frmEnd = 1560
    MOGthreshold = 190

elif dataset == 'traffic':
    tr_frmStart = 950
    tr_frmEnd = 1000
    te_frmStart = 1001
    te_frmEnd = 1050
    MOGthreshold = 330
else:
    print "You haven't defined the right dataset. Options are: highway, fall or traffic."
    exit(0)


mog2BG = cv2.BackgroundSubtractorMOG2(history=150, varThreshold=MOGthreshold, bShadowDetection = False)

groundTruthImgs = readGT(groundTruthPath, te_frmStart, te_frmEnd)

#########
# Task1 #
#########

# Hole filling
task1(mog2BG, inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dataset)


#########
# Task2 #
#########

# Area filtering
task2(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dataset)


#########
# Task3 #
#########

# Additional morphological operations
task3(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dataset)


#########
# Task4 #
#########

# Shadow removal
task4(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dataset)


#########
# Task5 #
#########

# Update PR Curve and AUC
task5(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dataset)
