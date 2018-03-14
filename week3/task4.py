from scipy import ndimage
from utils import *


def task4(inputpath, groundTruthPath, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dataset, thr, dimension, method='MOG'):
    pathDet = 'week3Results/task4/shadowDetection/' + dataset + '/'
    pathRem = 'week3Results/task4/shadowRemoval/' + dataset + '/'
    pathHol = 'week3Results/task4/holeFilling/' + dataset + '/'

    groundTruthImgs = readGT(groundTruthPath, te_frmStart, te_frmEnd, shadow=True)

    ImgNames = os.listdir(inputpath)
    ImgNames.sort()
    im = cv2.cvtColor(cv2.imread(inputpath + ImgNames[0]), cv2.COLOR_BGR2GRAY)

    cnn = 4
    hole = True

    if cnn == 4:
        str_elem = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    else:
        str_elem = np.ones((3, 3))

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
        bgad_hf = np.zeros((te_frmEnd - te_frmStart + 1, im.shape[0], im.shape[1]))
        bgad_ns = np.zeros((te_frmEnd - te_frmStart + 1, im.shape[0], im.shape[1]))

        for idx, name in enumerate(ImgNames):
            if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg'):
                if int(name[-8:-4]) >= tr_frmStart and int(name[-8:-4]) <= tr_frmEnd:
                    # Learning
                    out = mog2BG.apply(cv2.imread(inputpath + name), learningRate=0.01)

        te_ind = 0
        for idx, name in enumerate(ImgNames):
            if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg'):
                if int(name[-8:-4]) >= te_frmStart and int(name[-8:-4]) <= te_frmEnd:
                    # Testing
                    im = mog2BG.apply(cv2.imread(inputpath + name), learningRate=0.01)
                    cv2.imwrite(pathDet + name, im)
                    im[im == 127] = 50
                    bgad[te_ind, ...] = im

                    if hole:
                        # Shadow removal
                        im[im == 50] = 0
                        bgad_ns[te_ind, ...] = im/255
                        cv2.imwrite(pathRem + name, im)
                        # Apply Hole-Filling
                        bgad_hf[te_ind, ...] = ndimage.binary_fill_holes(im/255, str_elem).astype(int)
                        cv2.imwrite(pathHol + name, bgad_hf[te_ind, ...]*255)

                    te_ind += 1

    TP, TN, FP, FN = added_evaluation(groundTruthImgs, bgad, shadow=True)
    Recall, Pres, F1 = metrics(TP, TN, FP, FN)

    print dataset + ' dataset with ' + method + ' with shadows'
    print 'Recall: ' + str(Recall) + ' - Precision: ' + str(Pres) + ' - F1-score: ' + str(F1)
    print 'TASK4 finished'

    #"""
    groundTruthImgs = readGT(groundTruthPath, te_frmStart, te_frmEnd, shadow=False)

    TP, TN, FP, FN = added_evaluation(groundTruthImgs, bgad_ns, shadow=False)
    Recall, Pres, F1 = metrics(TP, TN, FP, FN)

    print dataset + ' dataset with ' + method + ' without shadows'
    print 'Recall: ' + str(Recall) + ' - Precision: ' + str(Pres) + ' - F1-score: ' + str(F1)
    print 'TASK4 finished'

    TP, TN, FP, FN = added_evaluation(groundTruthImgs, bgad_hf, shadow=False)
    Recall, Pres, F1 = metrics(TP, TN, FP, FN)

    print dataset + ' dataset with ' + method + ' without shadows and hole filling'
    print 'Recall: ' + str(Recall) + ' - Precision: ' + str(Pres) + ' - F1-score: ' + str(F1)
    print 'TASK4 finished'
    #"""
