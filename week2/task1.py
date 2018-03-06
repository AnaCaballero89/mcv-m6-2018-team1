import sys
sys.path.append('../')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import *

def task1():
    inputpath='../datasets/highway/input/'
    groundTruthPath='../datasets/highway/groundtruth/'

    frmStart=1050
    frmEnd=1099
    groundTruthImgs = readGT(groundTruthPath,1050+50,frmEnd+50)

    alpha=0.1
    alpharecall=[]
    alphaprec=[]
    alphaf1=[]
    alphalst=[]
    while alpha<=5:
        print alpha
        gauss=getGauss(inputpath,1050, 1099)
        bg=getBG(inputpath,frmStart+50,frmEnd+50,gauss, alpha)
        TestAmetric = []
        TestATP = []
        TestBTP = []
        TestAFN = []
        TestBFN = []
        TestTotalFG=[]
        TP_fnA=0
        TN_fnA=0
        FP_fnA=0
        FN_fnA=0


        for idx, img in enumerate(groundTruthImgs):
            pred_labels = bg[idx,:,:]
            true_labels=groundTruthImgs[idx,:,:]
            TP, TN, FP, FN = evaluation(pred_labels,true_labels)
            TestATP.append(TP)
            TP_fnA+=TP
            TN_fnA+=TN
            FP_fnA+=FP
            FN_fnA+=FN
            TestTotalFG.append(TP+FN)
            TestAmetric.append(metrics(TP, TN, FP, FN))

        alphaf1.append(metrics(TP_fnA, TN_fnA, FP_fnA, FN_fnA)[2])
        alpharecall.append(metrics(TP_fnA, TN_fnA, FP_fnA, FN_fnA)[0])
        alphaprec.append(metrics(TP_fnA, TN_fnA, FP_fnA, FN_fnA)[1])

        alphalst.append(alpha)
        alpha+=0.1
    a=np.asarray(TestAmetric)

    fig = plt.figure(figsize=(10, 5))
    #plt.axis([0, len(alphaf1), 0, 1])
    plt.title('Alpha vs Metrics')
    plt.plot(alphalst,alphaf1, c='r', label='F1')
    plt.plot(alphalst,alpharecall, c='b', label='Recall')
    plt.plot(alphalst,alphaprec, c='g', label='Precision')
    #plt.plot(b[:,2], c='r', label='Test B')
    plt.legend(loc='lower right');
    plt.xlabel('Alpha')
    plt.ylabel('Metrics')
    #plt.tight_layout()
    #plt.show()
    plt.savefig('F1_2.png')
    plt.close()
    pass
