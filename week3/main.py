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


def main():
    dataset = 'fall'  # 'fall', 'highway', 'traffic'

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

    if "2" in cv2.__version__:
        mog2BG = cv2.BackgroundSubtractorMOG2(history=150, varThreshold=MOGthreshold, bShadowDetection = False)
    else:
        mog2BG = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=MOGthreshold, detectShadows=False)
    groundTruthImgs = readGT(groundTruthPath, te_frmStart, te_frmEnd)


    #########
    # Task1 #
    #########

    # Hole filling
    #task1(mog2BG, inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dataset)


    #########
    # Task2 #
    #########
    """
    # Area filtering
    precLst=[]
    recLst=[]
    f1Lst=[]
    treshLst=[]
    AUC_lst=[]
    arthreshLst=[]
    conn,arthresh=4,0
    while arthresh<=1000:
        varThreshold=0
        while varThreshold<2000:
            if "2" in cv2.__version__:
                mog2BG = cv2.BackgroundSubtractorMOG2(history=150, varThreshold=MOGthreshold, bShadowDetection=False)
            else:
                mog2BG = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=MOGthreshold, detectShadows=False)
            groundTruthImgs = readGT(groundTruthPath, te_frmStart, te_frmEnd)
    
            #########
            # Task1 #
            #########
    
            # Hole filling
            recall, prec, f1=task2(mog2BG, inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dataset,conn,arthresh)
            precLst.append(prec)
            recLst.append(recall)
            f1Lst.append(f1)
            treshLst.append(treshLst)
            if recall<=0.01:
                break
            print varThreshold
            varThreshold+=100
        AUC0=skmetrics.auc(recLst, precLst, reorder=True)
        print 'AUC=', AUC0
        arthreshLst.append(arthresh)
        AUC_lst.append(AUC0)
        arthresh+=100
    print AUC_lst


    #########
    # Task3 #
    #########
    """
    # Additional morphological operations
    task3(mog2BG, inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dataset, MOGthreshold)
    
    
    #########
    # Task4 #
    #########
    """
    # Shadow removal
    task4(inputpath, groundTruthPath, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dataset, MOGthreshold, dimension=3, method='MOG')
    
    #########
    # Task5 #
    #########
    
    # Update PR Curve and AUC
    #task5(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dataset)
    """

main()