import sys
sys.path.append('../')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import *
import shutil
import ctypes as C
libmog = C.cdll.LoadLibrary('libmog2.so')

def getFG(img):
    (rows, cols) = (img.shape[0], img.shape[1])
    res = np.zeros(dtype=np.uint8, shape=(rows, cols))
    libmog.getfg(img.shape[0], img.shape[1],
                       img.ctypes.data_as(C.POINTER(C.c_ubyte)),
                       res.ctypes.data_as(C.POINTER(C.c_ubyte)))
    return res

def getBG(img):
    (rows, cols) = (img.shape[0], img.shape[1])
    res = np.zeros(dtype=np.uint8, shape=(rows, cols, 3))

    libmog.getbg(rows, cols, res.ctypes.data_as(C.POINTER(C.c_ubyte)))
    return res

def createTmpSequence(frmStart, frmEnd, choiceOfDataset):

    # Copy files
    for frameCounter in range(frmStart, frmEnd+1):
        if choiceOfDataset == 'traffic' and frameCounter < 1000:
            shutil.copy('../datasets/'+choiceOfDataset+'/input/in000'+str(frameCounter)+'.jpg', '../datasets/'+choiceOfDataset+'/tmpSequence')
        else:
            shutil.copy('../datasets/'+choiceOfDataset+'/input/in00'+str(frameCounter)+'.jpg', '../datasets/'+choiceOfDataset+'/tmpSequence')

    # Rename files
    counter = 0
    filenames = os.listdir("../datasets/"+choiceOfDataset+"/tmpSequence/")
    filenames.sort()
    for filename in filenames:
        filename = '../datasets/'+choiceOfDataset+'/tmpSequence/'+filename
        # print filename
        if counter < 10:
            os.rename(filename, '../datasets/'+choiceOfDataset+'/tmpSequence/in00'+str(counter)+'.jpg')
        elif counter < 100:
            os.rename(filename, '../datasets/'+choiceOfDataset+'/tmpSequence/in0'+str(counter)+'.jpg')
        else:
            os.rename(filename, '../datasets/'+choiceOfDataset+'/tmpSequence/in'+str(counter)+'.jpg')
        counter += 1
    return

def evaluation2(pred_labels,true_labels,frame,tst):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

    rgbArray = np.zeros((pred_labels.shape[0],pred_labels.shape[1],3), 'uint8')
    r=np.logical_and(pred_labels == 1, true_labels == 0).astype(int)
    g=np.logical_and(pred_labels == 1, true_labels == 1).astype(int)
    b=np.logical_and(pred_labels == 0, true_labels == 1).astype(int)
    rgbArray[..., 0] = r*255
    rgbArray[..., 1] = g*255
    rgbArray[..., 2] = b*255
    im = Image.fromarray(rgbArray)
    im.save(tst+'bg_'+str(frame)+'.jpg')
    return TP, TN, FP, FN

def task3():
    choiceOfDataset = 'fall'     # Possible options: 'highway', 'fall' and 'traffic'

    # Reset tmp folders
    shutil.rmtree('../datasets/'+choiceOfDataset+'/tmpSequence/')
    os.makedirs('../datasets/'+choiceOfDataset+'/tmpSequence')
    shutil.rmtree('task2Results/FG_evaluation/')
    os.makedirs('task2Results/FG_evaluation')
    shutil.rmtree('task2Results/FG_evaluation2/')
    os.makedirs('task2Results/FG_evaluation2')

    if choiceOfDataset == 'highway':
        frmStart = 1050
        frmEnd = 1200
    elif choiceOfDataset == 'fall':
        frmStart = 1460
        frmEnd = 1510
    elif choiceOfDataset == 'traffic':
        frmStart = 950
        frmEnd = 1000

    createTmpSequence(frmStart, frmEnd, choiceOfDataset)
    mogBG = cv2.BackgroundSubtractorMOG(history=150, nmixtures=5, backgroundRatio=0.0001)
    mog2BG = cv2.BackgroundSubtractorMOG2(history=150, varThreshold=150, bShadowDetection = False)

    cap = cv2.VideoCapture('../datasets/'+choiceOfDataset+'/tmpSequence/in%03d.jpg')
    ret, frame = cap.read()


    frameCounter = frmStart
    # While we have not reached the end of the sequence
    while frame is not None:
        # Show and write frames
        # cv2.imshow('frame', frame)
        cv2.imwrite('task2Results/frame/fr'+str(frameCounter)+'.jpg', frame)
        # Show and write foreground
        FG = getFG(frame)
        FG = cv2.threshold(FG, 50, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow('FG', FG)
        cv2.imwrite('task2Results/FG/in00'+str(frameCounter)+'.jpg', FG/255)
        # Show and write background
        BG = getBG(frame)
        # cv2.imshow('BG', BG)
        cv2.imwrite('task2Results/BG/bg'+str(frameCounter)+'.jpg', BG)
        if cv2.waitKey(1) == 27:
            exit(0)

        # Learn the background
        out = mogBG.apply(frame, learningRate=0.1)
        out2 = mog2BG.apply(frame, learningRate=0.01)
        cv2.imwrite('task2Results/FGout_01/fg'+str(frameCounter)+'.jpg', out)
        cv2.imwrite('task2Results/FGout2_001/fg'+str(frameCounter)+'.jpg', out2)
        # cv2.imshow('out', out)
        # cv2.imshow('out2', out2)

        frameCounter += 1
        # Read next frame
        _, frame = cap.read()

    #############################
    #        EVALUATION         #
    #############################

    allFramesResults = []

    # Reset tmp folder
    shutil.rmtree('../datasets/'+choiceOfDataset+'/tmpSequence/')
    os.makedirs('../datasets/'+choiceOfDataset+'/tmpSequence')

    if choiceOfDataset == 'highway':
        frmStart = 1200
        frmEnd = 1350
    elif choiceOfDataset == 'fall':
        frmStart = 1510
        frmEnd = 1560
    elif choiceOfDataset == 'traffic':
        frmStart = 1000
        frmEnd = 1050

    createTmpSequence(frmStart, frmEnd, choiceOfDataset)

    cap = cv2.VideoCapture('../datasets/'+choiceOfDataset+'/tmpSequence/in%03d.jpg')
    _, frame = cap.read()

    # While we have not reached the end of the sequence
    while frame is not None:

        # Learn the background
        out = mogBG.apply(frame, learningRate=0.1)
        out2 = mog2BG.apply(frame, learningRate=0.01)
        out = cv2.threshold(out, 50, 255, cv2.THRESH_BINARY)[1]
        out2 = cv2.threshold(out2, 50, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite('task2Results/FG_evaluation/in00'+str(frameCounter)+'.jpg', out/255.0)
        cv2.imwrite('task2Results/FG_evaluation2/in00'+str(frameCounter)+'.jpg', out2/255.0)
        frameCounter += 1

        # Read next frame
        _, frame = cap.read()


    # CHANGE NEXT LINE TO abSequencePath = 'task2Results/FG_evaluation2/' if you want to evaluate MOG2
    abSequencePath = 'task2Results/FG_evaluation2/'
    imgNames = os.listdir(abSequencePath)
    imgNames.sort()
    for name in imgNames:
        allFramesResults.append(cv2.cvtColor(cv2.imread(abSequencePath+name), cv2.COLOR_BGR2GRAY))
    allFramesResults = np.asarray(allFramesResults)

    metric = []
    TestTP = []
    TestFN = []
    TestTotalFG = []
    TP_res=0
    TN_res=0
    FP_res=0
    FN_res=0
    groundTruthImgs = readGT('../datasets/'+choiceOfDataset+'/groundtruth/', frmStart, frmEnd)

    for idx, img in enumerate(groundTruthImgs):
        pred_labels = allFramesResults[idx]
        true_labels = groundTruthImgs[idx,:,:]
        TP, TN, FP, FN = evaluation(pred_labels,true_labels)
        TestTP.append(TP)
        TP_res+=TP
        TN_res+=TN
        FP_res+=FP
        FN_res+=FN
        TestTotalFG.append(TP+FN)
        metric.append(metrics(TP, TN, FP, FN))
    recall, prec, f1 = metrics(TP_res, TN_res, FP_res, FN_res)

    print '\nMOG: results for '+choiceOfDataset+'\n\n Recall \t\t Precision \t\t F1\n', " %.3f" % recall, "\t\t\t", " %.3f" % prec, "\t\t\t", " %.3f" % f1

    # Reset tmp folder
    shutil.rmtree('../datasets/'+choiceOfDataset+'/tmpSequence/')
    os.makedirs('../datasets/'+choiceOfDataset+'/tmpSequence')
