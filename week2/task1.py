import sys
sys.path.append('../week1/')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from opticalFlowMetrics import opticalFlowMetrics
from utils import *

def getGauss(indir,frmStart, frmEnd):
    ImgNames = os.listdir(indir)
    ImgNames.sort()
    im=cv2.cvtColor(cv2.imread(indir+ImgNames[0]), cv2.COLOR_BGR2GRAY)
    gauss = np.zeros((im.shape[0],im.shape[1],frmEnd-frmStart+1), 'uint8')

    i=0
    for idx, name in enumerate(ImgNames):
        if int(name[-8:-4]) >= frmStart and int(name[-8:-4]) <= frmEnd:
            #im=cv2.cvtColor(cv2.imread(indir+name), cv2.COLOR_BGR2HSV)
            im=cv2.cvtColor(cv2.imread(indir+name), cv2.COLOR_BGR2GRAY)
            gauss[..., i]=im
            i+=1
    return gauss.mean(axis=2),gauss.std(axis=2)

def getBG(indir,frmStart, frmEnd,gauss, alpha=1,outdir=None):
    ImgNames = os.listdir(indir)
    ImgNames.sort()
    BGimgs=[]
    nms=[]
    for idx, name in enumerate(ImgNames):
        if int(name[-8:-4]) >= frmStart and int(name[-8:-4]) <= frmEnd:
            im=cv2.cvtColor(cv2.imread(indir+name), cv2.COLOR_BGR2GRAY)
            bg=(abs(im-gauss[0])>=alpha*(gauss[1]+2)).astype(int)
            BGimgs.append(bg)
            if outdir is not None:
                im = Image.fromarray((bg*255).astype('uint8'))
    
                im.save(outdir+name)
    return np.asarray(BGimgs)

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
        bg=getBG(inputpath,frmStart+50,frmEnd+50,gauss, alpha,tstbg)
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
            TP, TN, FP, FN = evaluation(pred_labels,true_labels,idx,'A')
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
