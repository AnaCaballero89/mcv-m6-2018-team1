import sys
sys.path.append('../')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import *
from task1 import task1
from task2 import task2
#from task3 import task3
#from task4 import task4

dataset = 'highway'  # 'fall', 'highway', 'traffic'

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

dimension = 1

#########
# Task1 #
#########

# Gaussian distribution + evaluation
task1(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dimension=dimension, grid_search=False, dataset=dataset)


#########
# Task2 #
#########

# Recursive Gaussian modeling + Evaluate and comparison to non-recursive
task2(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dimension, grid_search=False, dataset=dataset)


#########
# Task3 #
#########

# Comparison with state-of-the-art
task3(dataset, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd)


#########
# Task4 #
#########

# Color sequences
task4(choiceOfDataset=dataset, frmStartTr=tr_frmStart, frmEndTr=tr_frmEnd)
