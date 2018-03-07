import sys
sys.path.append('../')
import numpy as np
from utils import *
import matplotlib.pyplot as plt


def task2(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, grid_search=False, dataset='highway'):

    print 'TASK2 in process'

    if dataset == 'highway':
        alpha = 3.
        rho = .3
    elif dataset == 'fall':
        alpha = 0.
        rho = .0
    elif dataset == 'traffic':
        alpha = 0.
        rho = .0

    if grid_search:
        alpha, rho = get_alpha_rho(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd)

    gauss = getGauss(inputpath, tr_frmStart, tr_frmEnd)

    # Adaptive model
    bgad = getBG(inputpath, te_frmStart, te_frmEnd, gauss, alpha, rho, adaptive=True)

    TP, TN, FP, FN = added_evaluation(groundTruthImgs, bgad)
    Recall, Pres, F1 = metrics(TP, TN, FP, FN)

    print dataset + ' dataset with adaptive modelling'
    print 'Recall: ' + str(Recall) + ' - Precision: ' + str(Pres) + ' - F1-score: ' + str(F1)
    print 'Best alpha: ' + str(alpha)
    print 'Best rho: ' + str(rho)
    print 'TASK2 finished'
