import sys
sys.path.append('../')
from utils import *


def task2(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dimension=1, grid_search=False, dataset='highway'):

    print 'TASK2 in process'

    if dataset == 'highway':
        alpha = 4.6
        rho = .2
    elif dataset == 'fall':
        alpha = 5.
        rho = .1
    elif dataset == 'traffic':
        alpha = 5.
        rho = .2

    if grid_search:
        alpha, rho = get_alpha_rho(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dataset)

    gauss = getGauss(inputpath, tr_frmStart, tr_frmEnd, dimension=dimension)

    # Adaptive model
    bgad = getBG(inputpath, te_frmStart, te_frmEnd, gauss, alpha, rho, adaptive=True, dimension=dimension, outdir='bgImages/'+dataset+'/')

    TP, TN, FP, FN = added_evaluation(groundTruthImgs, bgad)
    Recall, Pres, F1 = metrics(TP, TN, FP, FN)

    print dataset + ' dataset with adaptive modelling'
    print 'Recall: ' + str(Recall) + ' - Precision: ' + str(Pres) + ' - F1-score: ' + str(F1)
    print 'Best alpha: ' + str(alpha)
    print 'Best rho: ' + str(rho)
    print 'TASK2 finished'
