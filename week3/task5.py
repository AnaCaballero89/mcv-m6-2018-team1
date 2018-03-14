import numpy as np
import cv2
from task3 import task3
from task4 import task4
import os
import shutil
from scipy import ndimage
from utils import createTmpSequence, createMOG, evaluation, metrics, roc, arfilt

def task5(mog2BG, inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, choiceOfDataset, MOGthreshold, groundTruthPath):

    # Create folders for the output or reset them if they already exist
    if not os.path.isdir('week3Results/task5/'):
        os.makedirs('week3Results/task5/')

    lst_tempdir=['week3Results/task5/shadowOutput/' + choiceOfDataset + '/',
                 'week3Results/task5/FG_evaluation/' + choiceOfDataset + '/']

    for dirs in lst_tempdir:
        if os.path.isdir(dirs):
            shutil.rmtree(dirs)
        os.makedirs(dirs)


    # Define structuring elements
    if choiceOfDataset == 'highway':
        structelement = np.ones((5, 5), np.uint8)
        structelement2 = structelement

    elif choiceOfDataset == 'fall':

        structelement = np.ones((7, 7), np.uint8)
        structelement2 = np.ones((3, 3), np.uint8)



    elif choiceOfDataset == 'traffic':
        structelement = np.ones((3, 3), np.uint8)


        structelement2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],dtype=np.uint8)


        structelement3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                                      [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                      [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                      [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],dtype=np.uint8)



    mog2BG = createMOG(hist=150, thr=MOGthreshold, shadows=True)

    # task4(inputpath, groundTruthPath, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, choiceOfDataset, MOGthreshold, dimension=3, method='MOG')

    createTmpSequence(tr_frmStart, tr_frmEnd, choiceOfDataset)

    cap = cv2.VideoCapture('../datasets/'+choiceOfDataset+'/tmpSequence/in%03d.jpg')
    ret, frame = cap.read()

    frameCounter = tr_frmStart

    # While we have not reached the end of the sequence
    while frame is not None:

        im = mog2BG.apply(frame, learningRate=0.01)
        im[im == 127] = 0
        # cv2.imshow('',im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(lst_tempdir[0]+'/fg'+str(frameCounter)+'.jpg', im)

        frameCounter += 1
        # Read next frame
        _, frame = cap.read()

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
        out[out == 127] = 0

        # Apply Morphological Operators

        if choiceOfDataset == 'highway':

            out = ndimage.binary_fill_holes(out).astype(np.float32)
            out = cv2.erode(out, structelement2, iterations=1)
            out = cv2.dilate(out, structelement2, iterations=1)
            out = cv2.dilate(out, structelement, iterations=1)
            out = cv2.erode(out, structelement, iterations=1)
            out = ndimage.binary_fill_holes(out).astype(np.float32)

        elif choiceOfDataset == 'fall':
            out = arfilt(out,4,300)
            out = ndimage.binary_fill_holes(out).astype(np.float32)
            out = cv2.erode(out, structelement2, iterations=1)
            out = cv2.dilate(out, structelement2, iterations=1)
            out = cv2.dilate(out, structelement, iterations=1)
            out = cv2.erode(out, structelement, iterations=1)
            out = ndimage.binary_fill_holes(out).astype(np.float32)


        elif choiceOfDataset == 'traffic':
            # out = arfilt(out,4,300)
            out = ndimage.binary_fill_holes(out).astype(np.float32)
            out = cv2.erode(out, structelement2, iterations=1)
            out = cv2.dilate(out, structelement2, iterations=1)
            #out = cv2.erode(out, structelement, iterations=1)
            #out = cv2.dilate(out, structelement, iterations=1)
            out = cv2.dilate(out, structelement3, iterations=1)
            out = cv2.erode(out, structelement3, iterations=1)
            out = ndimage.binary_fill_holes(out).astype(np.float32)
            # out = arfilt(out,4,300)

        cv2.imwrite('week3Results/task5/FG_evaluation/'+choiceOfDataset+'/in00'+str(frameCounter)+'.jpg', out*255)
        frameCounter += 1

        # Read next frame
        _, frame = cap.read()


    #############################
    #        EVALUATION         #
    #############################
    allFramesResults = []
    recallList = []
    precisionList = []
    sequencePath = 'week3Results/task5/FG_evaluation/'+choiceOfDataset+'/'
    imgNames = os.listdir(sequencePath)
    imgNames.sort()
    for name in imgNames:
        allFramesResults.append(cv2.cvtColor(cv2.imread(sequencePath+name), cv2.COLOR_BGR2GRAY))
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

    print '\nTask3: Morphological Operators'
    print "MOG: results for "+choiceOfDataset+" with threshold = "+str(MOGthreshold)+"\n\n Recall \t\t Precision \t\t F1\n", " %.3f" % recall, "\t\t\t", " %.3f" % prec, "\t\t\t", " %.3f" % f1

    recallList.append(recall)
    precisionList.append(prec)
    # Reset tmp folder
    shutil.rmtree('../datasets/'+choiceOfDataset+'/tmpSequence/')
    os.makedirs('../datasets/'+choiceOfDataset+'/tmpSequence')

    # roc(recallList, precisionList, 'ROC for MOG2')


    print "\nTask5 - Done"
