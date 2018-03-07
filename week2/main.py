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

dataset='fall' # 'highway', 'traffic'

inputpath='../datasets/' + dataset + '/input/'
groundTruthPath='../datasets/' + dataset + '/groundtruth/'

if dataset == 'highway':
    tr_frmStart = 1050
    tr_frmEnd = 1200
    te_frmStart = 1201
    te_frmEnd = 1350

elif dataset == 'fall':
    tr_frmStart = 1460
    tr_frmEnd = 1510
    te_frmStart = 1511
    te_frmEnd = 1560

elif dataset == 'traffic':
    tr_frmStart = 950
    tr_frmEnd = 1000
    te_frmStart = 1001
    te_frmEnd = 1050

groundTruthImgs = readGT(groundTruthPath, te_frmStart, te_frmEnd)

#########
# Task1 #
#########

# Gaussian distribution + evaluation
print 'aca'
task1(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, grid_search=True)


#########
# Task2 #
#########

# Recursive Gaussian modeling + Evaluate and comparison to non-recursive
task2(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, grid_search=True, dataset=dataset)


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
