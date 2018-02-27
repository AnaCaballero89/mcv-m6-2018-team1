
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def evaluation(pred_labels,true_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    return TP, TN, FP, FN


def metrics(TP, TN, FP, FN):
    Pres=float(TP)/(TP+FP+1e-10)
    Recall=float(TP)/(TP+FN+1e-10)
    F1=2*(Pres*Recall)/(Pres+Recall+1e-10)
    return Recall, Pres, F1


def readTest(abSequencePath):
    imgNames = os.listdir(abSequencePath)
    imgNames.sort()
    AImgs = []
    BImgs = []

    for name in imgNames:
        if 'A' in name:
            AImgs.append(cv2.cvtColor(cv2.imread(abSequencePath+name), cv2.COLOR_BGR2GRAY))
        elif 'B' in name:
            BImgs.append(cv2.cvtColor(cv2.imread(abSequencePath+name), cv2.COLOR_BGR2GRAY))

    AImgs = np.asarray(AImgs)
    BImgs = np.asarray(BImgs)
    return AImgs, BImgs


def readGT(groundTruthPath):
    groundTruthImgNames = os.listdir(groundTruthPath)
    groundTruthImgNames.sort()
    groundTruthImgs = []
    for name in groundTruthImgNames:
        if int(name[-8:-4]) > 1200 and int(name[-8:-4]) < 1401:
            im = cv2.threshold(cv2.cvtColor(cv2.imread(groundTruthPath+name), cv2.COLOR_BGR2GRAY), 250, 1, cv2.THRESH_BINARY)[1]
            groundTruthImgs.append(im)
    groundTruthImgs = np.asarray(groundTruthImgs)
    return groundTruthImgs

#########
# Task1 #
#########

# Implement the following metrics: Precision, Recall & F1-Score

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:21:35 2018

"""



# Read AB sequence images
abSequencePath = 'results_testAB_changedetection/results/highway/'
# Read ground truth images (from frame 1201 to frame 1400)
groundTruthPath = '../highway/groundtruth/'

AImgs, BImgs = readTest(abSequencePath)
groundTruthImgs = readGT(groundTruthPath)

TestAmetric = []
TestBmetric = []
TestATP = []
TestBTP = []
TestAFN = []
TestBFN = []
for idx, img in enumerate(groundTruthImgs):
    pred_labels = AImgs[idx,:,:]
    true_labels=groundTruthImgs[idx,:,:]
    TP, TN, FP, FN = evaluation(pred_labels,true_labels)
    TestATP.append(TP)
    TestAFN.append(FN)
    TestAmetric.append(metrics(TP, TN, FP, FN))
    pred_labels = BImgs[idx,:,:]
    TP, TN, FP, FN = evaluation(pred_labels,true_labels)
    TestBTP.append(TP)
    TestBFN.append(FN)
    TestBmetric.append(metrics(TP, TN, FP, FN))




#
#def calcmetric(im):
#    ima = cv2.imread(im)
#    thresh = 250
#    gt_bw = cv2.threshold(gtimg, thresh, 1, cv2.THRESH_BINARY)[1]
#
#    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
#    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
#    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
#    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
#    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
#    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
#    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

#########
# Task2 #
#########

# Temporal analysis:
# (True Positives and Total Foreground) vs time
# AND
# (F1 score) vs time



#########
# Task3 #
#########

# Optical flow evaluation metrics

flowResultFrame45 = cv2.imread('results_opticalflow_kitti/results/LKflow_000045_10.png')
flowResultFrame157 = cv2.imread('results_opticalflow_kitti/results/LKflow_000157_10.png')
flowGroundTruthFrame45 = cv2.imread('../optical_flow_w1task3/groundtruth/000045_10.png')
flowGroundTruthFrame157 = cv2.imread('../optical_flow_w1task3/groundtruth/000157_10.png')

pixelsResults45 = flowResultFrame45.flatten()
pixelsGT45 = flowGroundTruthFrame45.flatten()

# for pixelResult,pixelGT in zip(flowResultFrame45.flatten(), flowGroundTruthFrame45.flatten()):
    # print type(pixelResult)
    # print pixelResult
    # print np.shape(pixelResult)
    # if abs((float(pixelResult)-2**15/64.0)-(float(pixelGT)-2**15/64.0)) > 3:
        # print "New", pixelResult, pixelGT
        # print "difference", pixelResult-pixelGT




# cv2.imshow('',flowGroundTruthFrame157)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#########
# Task4 #
#########

# Desynchronized results



#########
# Task5 #
#########

# Visual representation optical flow
