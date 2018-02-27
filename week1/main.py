import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from opticalFlowMetrics import opticalFlowMetrics


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
            im = cv2.threshold(cv2.cvtColor(cv2.imread(groundTruthPath+name), cv2.COLOR_BGR2GRAY), 160, 1, cv2.THRESH_BINARY)[1]
            groundTruthImgs.append(im)
    groundTruthImgs = np.asarray(groundTruthImgs)
    return groundTruthImgs

def plotF1(a, b, fl=True):
    fig = plt.figure(figsize=(10, 5))
    plt.axis([0, len(a), 0, 1])
    plt.title('F1 Score vs Frame')
    plt.plot(a, c='b', label='Test A')
    plt.plot(b, c='r', label='Test B')
    plt.legend(loc='lower right')
    plt.xlabel('Frame')
    plt.ylabel('F1 Score')
    plt.tight_layout()
    if fl:
        plt.show()
    else:
        plt.savefig('F1.png')
    plt.close()


def plots(gt, a, b, fl=True):
    fig = plt.figure(figsize=(10, 5))
    plt.axis([0, len(gt), 0, max(gt) + 500])
    plt.title('True Positives and Total Foreground vs Frame')
    plt.plot(gt, linewidth=2, c='g', linestyle='--', label='Total Foreground')
    plt.plot(a, c='b', linewidth=1, label='Test A')
    plt.plot(b, c='r', linewidth=1, label='Test B')
    plt.legend(loc='lower right')
    plt.xlabel('Frame')
    plt.ylabel('True Positives')
    plt.tight_layout()
    if fl:
        plt.show()
    else:
        plt.savefig('TotalFG.png')
    plt.close()


def plot_desync(gt, A, name, step_lst, fl=True):
    fig = plt.figure(figsize=(10, 5))
    plt.axis([0, len(gt), 0, 1])
    plt.title(name + ' F1 Score Desync vs Frame')
    for s in step_lst:
        Testdes = desyncronization(gt, A, s)[:, 2]
        plt.plot(Testdes, label='Step %s' % s)
    plt.legend(loc='lower right')
    plt.xlabel('Frame')
    plt.ylabel('True Positives')
    plt.tight_layout()
    if fl:
        plt.show()
    else:
        plt.savefig('Des.png')
    plt.close()


def OFplots(ofImages, images):
    step = 10
    ind = 0

    for ofIm in ofImages:
        ofIm = cv2.resize(ofIm, (0, 0), fx=1. / step, fy=1. / step)
        rows, cols, depth = ofIm.shape
        U = []
        V = []

        for pixel in range(0, ofIm[:, :, 0].size):
            isOF = ofIm[:, :, 0].flat[pixel]
            if isOF == 1:
                U.append((((float)(ofIm[:, :, 1].flat[pixel]) - 2 ** 15) / 64.0) / 200.0)
                V.append((((float)(ofIm[:, :, 2].flat[pixel]) - 2 ** 15) / 64.0) / 200.0)
            else:
                U.append(0)
                V.append(0)

        U = np.reshape(U, (rows, cols))
        V = np.reshape(V, (rows, cols))
        x, y = np.meshgrid(np.arange(0, cols * step, step), np.arange(0, rows * step, step))

        plt.imshow(images[ind])
        plt.quiver(x, y, U, V, scale=0.1, alpha=1, color='r')
        plt.title('Optical Flow')
        plt.savefig('OF' + str(ind) + '.png')
        plt.show()
        plt.close()
        ind += 1

def desyncronization(gt, a, step=1):
    idx1 = 0
    idx2 = step
    TestDes = []
    while idx2 < gt.shape[0]:
        true_labels = gt[idx2, :, :]
        pred_labels = a[idx1, :, :]
        TP, TN, FP, FN = evaluation(pred_labels, true_labels)
        TestDes.append(metrics(TP, TN, FP, FN))

        idx1 += 1
        idx2 += 1

    TestDes = np.asarray(TestDes)
    return TestDes


def readOF(ofPath):
    imgNames = os.listdir(ofPath)
    imgNames.sort()
    images = []
    for name in imgNames:
        if name.endswith('.png'):
            # images.append(cv2.cvtColor(cv2.imread(ofPath+name), cv2.COLOR_BGR2GRAY))
            images.append(cv2.imread(ofPath+name, -1))
    return images


def readOFimages(ofOrPath):
    imgNames = os.listdir(ofOrPath)
    imgNames.sort()
    ofImages = []
    for name in imgNames:
        if name.endswith('.png'):
            if int(name[7:9]) == 10:
                im = cv2.imread(ofOrPath + name)
                ofImages.append(im)
    return ofImages


#########
# Task1 #
#########

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
