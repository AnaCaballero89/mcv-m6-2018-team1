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
    pass
