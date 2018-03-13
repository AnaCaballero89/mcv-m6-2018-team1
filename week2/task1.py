import sys
sys.path.append('../')
from utils import *


def task1(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dimension=1, grid_search=True, dataset='highway'):
    print 'TASK1 in process'

    if dataset == 'highway':
        alpha = 1.8
    elif dataset == 'fall':
        alpha = 4.3
    elif dataset == 'traffic':
        alpha = 1.8

    if grid_search:
        alpha = get_alpha_rho(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dataset, adaptive=False, dimension=dimension)[0]

    gauss = getGauss(inputpath, tr_frmStart, tr_frmEnd, dimension=dimension)
    bgad = getBG(inputpath, te_frmStart, te_frmEnd, gauss, alpha, adaptive=False, dimension=dimension)

    TP, TN, FP, FN = added_evaluation(groundTruthImgs, bgad)
    Recall, Pres, F1 = metrics(TP, TN, FP, FN)

    print dataset + ' dataset with non-adaptive modelling'
    print 'Recall: ' + str(Recall) + ' - Precision: ' + str(Pres) + ' - F1-score: ' + str(F1)
    print 'Best alpha: ' + str(alpha)
    print 'TASK1 finished'
