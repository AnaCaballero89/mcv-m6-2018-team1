import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from opticalFlowMetrics import opticalFlowMetrics
from utils import *


#########
# Task1 #
#########

# Read AB sequence images
abSequencePath = 'results_testAB_changedetection/results/highway/'
# Read ground truth images (from frame 1201 to frame 1400)
groundTruthPath = '../datasets/highway/groundtruth/'

AImgs, BImgs = readTest(abSequencePath)
groundTruthImgs = readGT(groundTruthPath)

TestAmetric = []
TestBmetric = []
TestATP = []
TestBTP = []
TestAFN = []
TestBFN = []
TestTotalFG=[]
TP_fnA=0
TN_fnA=0
FP_fnA=0
FN_fnA=0

TP_fnB=0
TN_fnB=0
FP_fnB=0
FN_fnB=0

for idx, img in enumerate(groundTruthImgs):
    pred_labels = AImgs[idx,:,:]
    true_labels=groundTruthImgs[idx,:,:]
    TP, TN, FP, FN = evaluation(pred_labels,true_labels)
    TestATP.append(TP)
    TP_fnA+=TP
    TN_fnA+=TN
    FP_fnA+=FP
    FN_fnA+=FN
    TestTotalFG.append(TP+FN)
    TestAmetric.append(metrics(TP, TN, FP, FN))
    pred_labels = BImgs[idx,:,:]
    TP, TN, FP, FN = evaluation(pred_labels,true_labels)
    TP_fnB+=TP
    TN_fnB+=TN
    FP_fnB+=FP
    FN_fnB+=FN
    TestBTP.append(TP)
    TestBmetric.append(metrics(TP, TN, FP, FN))
print 'testA', metrics(TP_fnA, TN_fnA, FP_fnA, FN_fnA)
print 'testB', metrics(TP_fnB, TN_fnB, FP_fnB, FN_fnB)


#########
# Task2 #
#########

# Temporal analysis:
# (True Positives and Total Foreground) vs time AND (F1 score) vs time
TestAmetric = np.asarray(TestAmetric)
TestBmetric = np.asarray(TestBmetric)

# Plot F1 score
# plotF1(TestAmetric[:, 2], TestBmetric[:, 2])

# Plot TP and TF
# plots(TestTotalFG, TestATP, TestBTP)


#########
# Task3 #
#########

# Optical flow evaluation metrics

flowResultFrame45 = cv2.imread('results_opticalflow_kitti/results/LKflow_000045_10.png',-1)
flowResultFrame157 = cv2.imread('results_opticalflow_kitti/results/LKflow_000157_10.png',-1)
flowGroundTruthFrame45 = cv2.imread('../optical_flow_w1task3/groundtruth/000045_10.png',-1)
flowGroundTruthFrame157 = cv2.imread('../optical_flow_w1task3/groundtruth/000157_10.png',-1)

# Calculate Optical Flow metrics
dist45 = (np.asarray(opticalFlowMetrics(flowResultFrame45, flowGroundTruthFrame45, 45)[2])).reshape((flowResultFrame45.shape[0],flowResultFrame45.shape[1]))
dist157 = (np.asarray(opticalFlowMetrics(flowResultFrame157, flowGroundTruthFrame157, 157)[2])).reshape((flowResultFrame157.shape[0],flowResultFrame157.shape[1]))

# Plotting #
dist45 = np.ma.masked_where(dist45 == 0,dist45)
cmap = plt.cm.RdYlBu_r
cmap.set_bad(color='black')

plt.imshow(dist45, interpolation='none', cmap=cmap)
plt.colorbar(orientation='horizontal')
plt.show()
plt.savefig('plot45.png')


dist157 = np.ma.masked_where(dist157 == 0,dist157)
cmap = plt.cm.RdYlBu_r
cmap.set_bad(color='black')

plt.imshow(dist157, interpolation='none', cmap=cmap)
plt.colorbar(orientation='horizontal')
plt.show()
plt.savefig('plot157.png')
plt.close()

#########
# Task4 #
#########

# Desynchronized results

step_lst = [0, 1, 3, 5, 10]
plot_desync(groundTruthImgs, AImgs, name='Test A', step_lst=step_lst)
plot_desync(groundTruthImgs, BImgs, name='Test B', step_lst=step_lst)


#########
# Task5 #
#########

# Visual representation optical flow

ofPath = 'results_opticalflow_kitti/results/'
ofOrPath = '../optical_flow_w1task3/'
ofImages = readOF(ofPath)
images = readOFimages(ofOrPath)

OFplots(ofImages, images)
