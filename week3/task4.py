from scipy import ndimage
from utils import *


def task4(inputpath, groundTruthPath, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dataset, thr, dimension, method='MOG'):
    groundTruthImgs = readGT(groundTruthPath, te_frmStart, te_frmEnd, shadow=True)

    ImgNames = os.listdir(inputpath)
    ImgNames.sort()
    im = cv2.cvtColor(cv2.imread(inputpath + ImgNames[0]), cv2.COLOR_BGR2GRAY)

    if method == 'Gaussian':
        if dataset == 'highway':
            alpha = 4.6
            rho = .2
        elif dataset == 'fall':
            alpha = 5.
            rho = .1
        elif dataset == 'traffic':
            alpha = 5.
            rho = .2

        gauss = getGauss(inputpath, tr_frmStart, tr_frmEnd, dimension=dimension)
        bgad = getBG(inputpath, te_frmStart, te_frmEnd, gauss, alpha, rho, adaptive=True, dimension=dimension)

    elif method == 'MOG':
        mog2BG = createMOG(hist=150, thr=thr, shadows=True)
        bgad = np.zeros((te_frmEnd - te_frmStart + 1, im.shape[0], im.shape[1]))

        for idx, name in enumerate(ImgNames):
            if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg'):
                if int(name[-8:-4]) >= tr_frmStart and int(name[-8:-4]) <= tr_frmEnd:
                    out = mog2BG.apply(cv2.imread(inputpath + name), learningRate=0.01)
                    # Apply Hole-Filling
                    #out = ndimage.binary_fill_holes(out).astype(int)

                    #tr_ind += 1
        te_ind = 0
        for idx, name in enumerate(ImgNames):
            if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg'):
                if int(name[-8:-4]) >= te_frmStart and int(name[-8:-4]) <= te_frmEnd:
                    im = mog2BG.apply(cv2.imread(inputpath + name), learningRate=0.01)
                    im[im == 127] = 50
                    bgad[te_ind, ...] = im
                    # Apply Hole-Filling
                    #out = ndimage.binary_fill_holes(out).astype(int)

                    te_ind += 1

    TP, TN, FP, FN = added_evaluation(groundTruthImgs, bgad, shadow=True)
    Recall, Pres, F1 = metrics(TP, TN, FP, FN)

    print dataset + ' dataset with ' + method
    print 'Recall: ' + str(Recall) + ' - Precision: ' + str(Pres) + ' - F1-score: ' + str(F1)
    print 'TASK4 finished'
