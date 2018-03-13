import sys
sys.path.append('../')
from scipy import ndimage
from utils import createTmpSequence, evaluation, metrics, roc, arfilt
import os
import shutil
import cv2
import numpy as np

def task2(mog2BG, inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, choiceOfDataset,conn,arthresh):

    recallList = []
    precisionList = []
    if not os.path.isdir('week3Results/task2/'):
        os.makedirs('week3Results/task2/')
    lst_tempdir=['week3Results/task2/FG_baselineWeek2_evaluation/',
                'week3Results/task2/FG_evaluation/',
                'week3Results/task2/holeFilling/',
                'week3Results/task2/FG_baselineWeek2_evaluation/']
    for dirs in lst_tempdir:
        
        if os.path.isdir(dirs):
            shutil.rmtree(dirs)
        os.makedirs(dirs)            
    # for threshold in range(10,400,20):
    # Reset tmp folders
    if 'tmpSequence' in os.listdir('../datasets/'+choiceOfDataset):
        shutil.rmtree('../datasets/'+choiceOfDataset+'/tmpSequence/')
    os.makedirs('../datasets/'+choiceOfDataset+'/tmpSequence')
    # if 'FG_evaluation' in os.listdir('../datasets/'+choiceOfDataset):
    #     shutil.rmtree('task3Results/FG_evaluation/')
    # os.makedirs('task3Results/FG_evaluation')

    createTmpSequence(tr_frmStart, tr_frmEnd, choiceOfDataset)
    
#    try:
#        mog2BG = cv2.BackgroundSubtractorMOG2(history=50, varThreshold=10, bShadowDetection = False)
#    except:
#        mog2BG = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=10)
    # Start reading the sequence
    cap = cv2.VideoCapture('../datasets/'+choiceOfDataset+'/tmpSequence/in%03d.jpg')
    ret, frame = cap.read()

    frameCounter = tr_frmStart
    # While we have not reached the end of the sequence
    while frame is not None:
        # Show and write frames
        # cv2.imwrite('task2Results/frame/fr'+str(frameCounter)+'.jpg', frame)
        # cv2.imwrite('task2Results/FG/in00'+str(frameCounter)+'.jpg', FG/255)

        # Learn the background
        out = mog2BG.apply(frame, learningRate=0.01)
        cv2.imwrite('week3Results/task1/baseWeek2Results/fg'+str(frameCounter)+'.jpg', out)

        # Apply Hole-Filling
        out = ndimage.binary_fill_holes(out).astype(int)
        cv2.imwrite('week3Results/task1/holeFilling/fg'+str(frameCounter)+'.jpg', out*255)
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

    createTmpSequence(te_frmStart, te_frmEnd, choiceOfDataset)

    cap = cv2.VideoCapture('../datasets/'+choiceOfDataset+'/tmpSequence/in%03d.jpg')
    _, frame = cap.read()

    # While we have not reached the end of the sequence
    while frame is not None:

        # Learn the background
        out = mog2BG.apply(frame, learningRate=0.01)
        baselineOut = cv2.threshold(out, 50, 255, cv2.THRESH_BINARY)[1]
        # Last week's best results
        cv2.imwrite('week3Results/task1/FG_baselineWeek2_evaluation/in00'+str(frameCounter)+'.jpg', baselineOut/255.0)
        out = ndimage.binary_fill_holes(out).astype(int)
        # Hole-filled images
        cv2.imwrite('week3Results/task1/FG_evaluation/in00'+str(frameCounter)+'.jpg', out)
        frameCounter += 1

        # Read next frame
        _, frame = cap.read()


    # CHANGE NEXT LINE TO sequencePath = 'week3Results/task1/FG_baselineWeek2_evaluation/' if you want to evaluate last week's best results
    sequencePath = 'week3Results/task1/FG_evaluation/'
    imgNames = os.listdir(sequencePath)
    imgNames.sort()
    for name in imgNames:
        inimg=cv2.cvtColor(cv2.imread(sequencePath+name), cv2.COLOR_BGR2GRAY)
        inimg=arfilt(inimg,conn,arthresh)
        allFramesResults.append(inimg)
    allFramesResults = np.asarray(allFramesResults)

    metric = []
    TestTP = []
    TestFN = []
    TestTotalFG = []
    TP_res=0
    TN_res=0
    FP_res=0
    FN_res=0

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

    print "\nMOG: results for "+choiceOfDataset+" Recall \t\t Precision \t\t F1\n", " %.3f" % recall, "\t\t\t", " %.3f" % prec, "\t\t\t", " %.3f" % f1

    recallList.append(recall)
    precisionList.append(prec)
    print recallList
    # Reset tmp folder
    shutil.rmtree('../datasets/'+choiceOfDataset+'/tmpSequence/')
    os.makedirs('../datasets/'+choiceOfDataset+'/tmpSequence')
    return recall, prec, f1
    # roc(recallList, precisionList, 'ROC for MOG2')
