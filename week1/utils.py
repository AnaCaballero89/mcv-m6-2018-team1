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


def readGT(groundTruthPath, frmStart=1201, frmEnd=1400):
    groundTruthImgNames = os.listdir(groundTruthPath)
    groundTruthImgNames.sort()
    groundTruthImgs = []
    for name in groundTruthImgNames:
        if int(name[-8:-4]) >= frmStart and int(name[-8:-4]) <= frmEnd:
            im = cv2.threshold(cv2.cvtColor(cv2.imread(groundTruthPath+name), cv2.COLOR_BGR2GRAY), 169, 1, cv2.THRESH_BINARY)[1]
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
