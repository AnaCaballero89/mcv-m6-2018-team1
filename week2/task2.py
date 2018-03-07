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

    if dimension == 1:
        gauss = getGauss(inputpath, tr_frmStart, tr_frmEnd)

        # Adaptive model
        bgad = getBG(inputpath, te_frmStart, te_frmEnd, gauss, alpha, rho, adaptive=True)

    else:
        gauss0 = getGaussRGB(inputpath, tr_frmStart, tr_frmEnd, 0)
        bg0 = getBGRGB(inputpath, te_frmStart, te_frmEnd, gauss0, 0, alpha)
        gauss1 = getGaussRGB(inputpath, tr_frmStart, tr_frmEnd, 1)
        bg1 = getBGRGB(inputpath, te_frmStart, te_frmEnd, gauss1, 1, alpha)
        gauss2 = getGaussRGB(inputpath, tr_frmStart, tr_frmEnd, 2)
        bg2 = getBGRGB(inputpath, te_frmStart, te_frmEnd, gauss2, 2, alpha)
        bgad = bg0 * bg1 * bg2
    TP, TN, FP, FN = added_evaluation(groundTruthImgs, bgad)
    Recall, Pres, F1 = metrics(TP, TN, FP, FN)

    print dataset + ' dataset with adaptive modelling'
    print 'Recall: ' + str(Recall) + ' - Precision: ' + str(Pres) + ' - F1-score: ' + str(F1)
    print 'Best alpha: ' + str(alpha)
    print 'Best rho: ' + str(rho)
    print 'TASK2 finished'
