import sys
sys.path.append('../')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from task3 import createTmpSequence
from utils import *
import ctypes as C
libmog = C.cdll.LoadLibrary('libmog2.so')


def task4(choiceOfDataset='highway', frmStartTr=1050, frmEndTr=1200, frmStartEv=1201, frmEndEv=1350):
    # Reset tmp folders
    shutil.rmtree('../datasets/'+choiceOfDataset+'/tmpSequence/')
    os.makedirs('../datasets/'+choiceOfDataset+'/tmpSequence')

    createTmpSequence(frmStartTr, frmEndTr, choiceOfDataset)

    cap = cv2.VideoCapture('../datasets/'+choiceOfDataset+'/tmpSequence/in%03d.jpg')
    ret, frame = cap.read()


    frameCounter = frmStartTr
    # While we have not reached the end of the sequence
    while frame is not None:

        # cv2.imwrite('task2Results/FG/in00'+str(frameCounter)+'.jpg', FG/255)
        print type(frame)
        frameChannel1 = frame[:,:,0]
        frameChannel2 = frame[:,:,1]
        frameChannel3 = frame[:,:,2]
        # print np.shape(frame)
        # frameChannel1,frameChannel2,frameChannel3 = cv2.split(frame)
        print frameChannel1[100,100]
        print frameChannel2[100,100]
        print frameChannel3[100,100]
        print "\n"
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # numpy_horizontal = np.hstack((frameChannel1, frameChannel2))
        numpy_horizontal_concat = np.concatenate((frameChannel1, frameChannel2,frameChannel3), axis=1)
        cv2.imshow('numpy_horizontal_concat', numpy_horizontal_concat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.imshow('Channel2', frameChannel2)
        # cv2.imshow('Channel3', frameChannel3)
        if cv2.waitKey(1) == 27:
            exit(0)

        frameCounter += 1

        # Read next frame
        _, frame = cap.read()
