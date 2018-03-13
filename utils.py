import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn import metrics as skmetrics
import pickle
import shutil
from skimage import measure


def evaluation(pred_labels, true_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    return TP, TN, FP, FN


def metrics(TP, TN, FP, FN):
    Pres=float(TP)/(TP+FP+1e-10)
    Recall=float(TP)/(TP+FN+1e-10)
    F1=2*(Pres*Recall)/(Pres+Recall+1e-10)
    return Recall, Pres, F1


def readTest(abSequencePath):
    imgNames = os.listdir(abSequencePath)
    imgNames.sort()
    AImgs = []
    BImgs = []

    for name in imgNames:
        if 'A' in name:
            AImgs.append(cv2.cvtColor(cv2.imread(abSequencePath+name), cv2.COLOR_BGR2GRAY))
        elif 'B' in name:
            BImgs.append(cv2.cvtColor(cv2.imread(abSequencePath+name), cv2.COLOR_BGR2GRAY))

    AImgs = np.asarray(AImgs)
    BImgs = np.asarray(BImgs)
    return AImgs, BImgs


def readGT(groundTruthPath, frmStart=1201, frmEnd=1400, shadow=False):
    groundTruthImgNames = os.listdir(groundTruthPath)
    groundTruthImgNames.sort()
    groundTruthImgs = []
    for name in groundTruthImgNames:
        if int(name[-8:-4]) >= frmStart and int(name[-8:-4]) <= frmEnd:
            if not shadow:
                im = cv2.threshold(cv2.cvtColor(cv2.imread(groundTruthPath+name), cv2.COLOR_BGR2GRAY), 169, 1, cv2.THRESH_BINARY)[1]
            else:
                im = cv2.cvtColor(cv2.imread(groundTruthPath + name), cv2.COLOR_BGR2GRAY)[1]
            groundTruthImgs.append(im)
    groundTruthImgs = np.asarray(groundTruthImgs)
    return groundTruthImgs


def arfilt(im, connect=4, area_thresh=1):
    im=(im>0).astype(int)
    labeled = measure.label(im, neighbors=connect)
    props = measure.regionprops(labeled)
    area=np.zeros((len(props)+1,1))
    for prop in props:
        area[prop.label]=prop.area
    fil_img=(area[labeled,0]>area_thresh).astype(int)
    return fil_img


def plotF1(a, b, fl=True):
    fig = plt.figure(figsize=(10, 5))
    plt.axis([0, len(a), 0, 1])
    plt.title('F1 Score vs Frame')
    plt.plot(a, c='b', label='Test A')
    plt.plot(b, c='r', label='Test B')
    plt.legend(loc='lower right')
    plt.xlabel('Frame')
    plt.ylabel('F1 Score')
    plt.tight_layout()
    if fl:
        plt.show()
    else:
        plt.savefig('F1.png')
    plt.close()


def plots(gt, a, b, fl=True):
    fig = plt.figure(figsize=(10, 5))
    plt.axis([0, len(gt), 0, max(gt) + 500])
    plt.title('True Positives and Total Foreground vs Frame')
    plt.plot(gt, linewidth=2, c='g', linestyle='--', label='Total Foreground')
    plt.plot(a, c='b', linewidth=1, label='Test A')
    plt.plot(b, c='r', linewidth=1, label='Test B')
    plt.legend(loc='lower right')
    plt.xlabel('Frame')
    plt.ylabel('True Positives')
    plt.tight_layout()
    if fl:
        plt.show()
    else:
        plt.savefig('TotalFG.png')
    plt.close()


def plot_desync(gt, A, name, step_lst, fl=True):
    fig = plt.figure(figsize=(10, 5))
    plt.axis([0, len(gt), 0, 1])
    plt.title(name + ' F1 Score Desync vs Frame')
    for s in step_lst:
        Testdes = desyncronization(gt, A, s)[:, 2]
        plt.plot(Testdes, label='Step %s' % s)
    plt.legend(loc='lower right')
    plt.xlabel('Frame')
    plt.ylabel('True Positives')
    plt.tight_layout()
    if fl:
        plt.show()
    else:
        plt.savefig('Des.png')
    plt.close()


def OFplots(ofImages, images):
    step = 10
    ind = 0

    for ofIm in ofImages:
        ofIm = cv2.resize(ofIm, (0, 0), fx=1. / step, fy=1. / step)
        rows, cols, depth = ofIm.shape
        U = []
        V = []

        for pixel in range(0, ofIm[:, :, 0].size):
            isOF = ofIm[:, :, 0].flat[pixel]
            if isOF == 1:
                U.append((((float)(ofIm[:, :, 1].flat[pixel]) - 2 ** 15) / 64.0) / 200.0)
                V.append((((float)(ofIm[:, :, 2].flat[pixel]) - 2 ** 15) / 64.0) / 200.0)
            else:
                U.append(0)
                V.append(0)

        U = np.reshape(U, (rows, cols))
        V = np.reshape(V, (rows, cols))
        x, y = np.meshgrid(np.arange(0, cols * step, step), np.arange(0, rows * step, step))

        plt.imshow(images[ind])
        plt.quiver(x, y, U, V, scale=0.1, alpha=1, color='r')
        plt.title('Optical Flow')
        plt.savefig('OF' + str(ind) + '.png')
        plt.show()
        plt.close()
        ind += 1


def desyncronization(gt, a, step=1):
    idx1 = 0
    idx2 = step
    TestDes = []
    while idx2 < gt.shape[0]:
        true_labels = gt[idx2, :, :]
        pred_labels = a[idx1, :, :]
        TP, TN, FP, FN = evaluation(pred_labels, true_labels)
        TestDes.append(metrics(TP, TN, FP, FN))

        idx1 += 1
        idx2 += 1

    TestDes = np.asarray(TestDes)
    return TestDes


def readOF(ofPath):
    imgNames = os.listdir(ofPath)
    imgNames.sort()
    images = []
    for name in imgNames:
        if name.endswith('.png'):
            # images.append(cv2.cvtColor(cv2.imread(ofPath+name), cv2.COLOR_BGR2GRAY))
            images.append(cv2.imread(ofPath+name, -1))
    return images


def readOFimages(ofOrPath):
    imgNames = os.listdir(ofOrPath)
    imgNames.sort()
    ofImages = []
    for name in imgNames:
        if name.endswith('.png'):
            if int(name[7:9]) == 10:
                im = cv2.imread(ofOrPath + name)
                ofImages.append(im)
    return ofImages


def getGauss(indir, frmStart, frmEnd, dimension=1):
    ImgNames = os.listdir(indir)
    ImgNames.sort()
    im = cv2.cvtColor(cv2.imread(indir + ImgNames[0]), cv2.COLOR_BGR2GRAY)

    if dimension == 1:
        gauss = np.zeros((im.shape[0], im.shape[1], frmEnd - frmStart + 1))
    else:
        gaussR = np.zeros((im.shape[0], im.shape[1], frmEnd - frmStart + 1))
        gaussG = np.zeros((im.shape[0], im.shape[1], frmEnd - frmStart + 1))
        gaussB = np.zeros((im.shape[0], im.shape[1], frmEnd - frmStart + 1))

    i = 0
    for idx, name in enumerate(ImgNames):
        if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg'):
            if int(name[-8:-4]) >= frmStart and int(name[-8:-4]) <= frmEnd:

                if dimension == 1:
                    im = cv2.cvtColor(cv2.imread(indir + name), cv2.COLOR_BGR2GRAY)
                    gauss[..., i] = im
                else:
                    im = cv2.imread(indir + name)
                    im = cv2.split(im)
                    gaussR[..., i] = im[0]
                    gaussG[..., i] = im[1]
                    gaussB[..., i] = im[2]

                i += 1

    if dimension == 1:
        mean = gauss.mean(axis=2)
        var = gauss.var(axis=2)
    else:
        mean = np.stack((gaussR.mean(axis=2), gaussG.mean(axis=2), gaussB.mean(axis=2)), axis=2)
        var = np.stack((gaussR.var(axis=2), gaussG.var(axis=2), gaussB.var(axis=2)), axis=2)

    return mean, var


def getBG(indir, frmStart, frmEnd, gauss, alpha=1, rho=0.1, outdir=None, adaptive=False, dimension=1, shadow=False):
    ImgNames = os.listdir(indir)
    ImgNames.sort()
    BGimgs = []
    for idx, name in enumerate(ImgNames):
        if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg'):
            if int(name[-8:-4]) >= frmStart and int(name[-8:-4]) <= frmEnd:

                if dimension == 1:
                    im = cv2.cvtColor(cv2.imread(indir + name), cv2.COLOR_BGR2GRAY)
                    bg = (abs(im - gauss[0]) >= alpha * (np.sqrt(gauss[1]) + 2)).astype(int)

                else:
                    im = cv2.imread(indir + name)
                    bg = (abs(im - gauss[0]) >= alpha * (np.sqrt(gauss[1]) + 2)).astype(int)
                    bg = np.logical_and(bg[..., 0], bg[..., 1], bg[..., 2])

                    if shadow:
                        im = cv2.split(im)
                        a_shadow = 1

                if adaptive:
                    gmean = rho*im + (1-rho)*gauss[0]
                    gvar = (rho*(im-gmean))**2 + (1-rho)*gauss[1]
                    gauss[0][bg == 0] = gmean[bg == 0]
                    gauss[1][bg == 0] = gvar[bg == 0]

                BGimgs.append(bg)
                if outdir is not None:
                    im = Image.fromarray((bg * 255).astype('uint8'))
                    im.save(outdir + name)

    return np.asarray(BGimgs)


def getGaussRGB(indir, frmStart, frmEnd, channel=0):
    ImgNames = os.listdir(indir)
    ImgNames.sort()
    im = cv2.cvtColor(cv2.imread(indir + ImgNames[0]), cv2.COLOR_BGR2GRAY)
    gauss = np.zeros((im.shape[0], im.shape[1], frmEnd - frmStart + 1))

    i = 0
    for idx, name in enumerate(ImgNames):
        if int(name[-8:-4]) >= frmStart and int(name[-8:-4]) <= frmEnd:
            # im=cv2.cvtColor(cv2.imread(indir+name), cv2.COLOR_BGR2HSV)
            im = cv2.imread(indir + name)
            im = cv2.split(im)
            im = im[channel]
            gauss[..., i] = im
            i += 1
    return gauss.mean(axis=2), gauss.std(axis=2)


def getBGRGB(indir, frmStart, frmEnd, gauss, channel=0, alpha=1, rho=0.1, outdir=None, adaptive=False):
    ImgNames = os.listdir(indir)
    ImgNames.sort()
    BGimgs = []
    for idx, name in enumerate(ImgNames):
        if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg'):
            if int(name[-8:-4]) >= frmStart and int(name[-8:-4]) <= frmEnd:
                im = cv2.imread(indir + name)
                im = cv2.split(im)
                im = im[channel]
                bg = (abs(im - gauss[0]) >= alpha * (gauss[1] + 2)).astype(int)

                if adaptive:
                    gmean = rho*im + (1-rho)*gauss[0]
                    gvar = (rho*(im-gmean))**2 + (1-rho)*gausss[1]
                    gauss[0][bg == 0] = gmean[bg == 0]
                    gauss[1][bg == 0] = gvar[bg == 0]

                BGimgs.append(bg)
                if outdir is not None:
                    im = Image.fromarray((bg * 255).astype('uint8'))
                    im.save(outdir + name)
    return np.asarray(BGimgs)


def annot_max(x, y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = "$\\alpha$={:.1f}, F1={:.3f}".format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.7, ymax + 0.2), **kw)


def added_evaluation(groundTruthImgs, bgad):
    TP_fnA = 0
    TN_fnA = 0
    FP_fnA = 0
    FN_fnA = 0

    for idx, img in enumerate(groundTruthImgs):
        pred_labels = bgad[idx, :, :]
        true_labels = groundTruthImgs[idx, :, :]
        TP, TN, FP, FN = evaluation(pred_labels, true_labels)
        TP_fnA += TP
        TN_fnA += TN
        FP_fnA += FP
        FN_fnA += FN

    return TP_fnA, TN_fnA, FP_fnA, FN_fnA


def get_alpha_rho(inputpath, groundTruthImgs, tr_frmStart, tr_frmEnd, te_frmStart, te_frmEnd, dataset, show_plt=False, adaptive=True, dimension=1):
    alpha_start = 0.1
    rho_start = 0.
    maxa = 5
    maxr = 0

    up = 0.1
    f1 = []

    inda = 0
    if adaptive:
        rho_start = 0.1
        maxr = 1
        indr = 0
        f1_mat = np.zeros((int(maxa / up), int(maxr / up)))
        pre_mat = np.zeros((int(maxa / up), int(maxr / up)))
        rec_mat = np.zeros((int(maxa / up), int(maxr / up)))

    else:
        TestATP = []
        TestAFN = []
        TestAFP = []
        TestATN = []
        prec = []
        recall = []
        indr = None
        f1_mat = np.zeros((int(maxa / up),))
        pre_mat = np.zeros((int(maxa / up), ))
        rec_mat = np.zeros((int(maxa / up), ))

    alpha = alpha_start
    rho = rho_start

    while alpha <= maxa:

        while rho <= maxr:
            print alpha, rho

            if dimension == 1:
                gauss = getGauss(inputpath, tr_frmStart, tr_frmEnd)

                # Adaptive model
                bgad = getBG(inputpath, te_frmStart, te_frmEnd, gauss, alpha, rho, adaptive=adaptive)

            else:
                gauss = getGauss(inputpath, tr_frmStart, tr_frmEnd, dimension=dimension)
                bgad = getBG(inputpath, te_frmStart, te_frmEnd, gauss, alpha, adaptive=False, dimension=dimension)

            # Adaptative variables
            TP_fnA, TN_fnA, FP_fnA, FN_fnA = added_evaluation(groundTruthImgs, bgad)
            f1.append(metrics(TP_fnA, TN_fnA, FP_fnA, FN_fnA)[2])

            pre_mat[inda, indr] = metrics(TP_fnA, TN_fnA, FP_fnA, FN_fnA)[1]
            rec_mat[inda, indr] = metrics(TP_fnA, TN_fnA, FP_fnA, FN_fnA)[0]
            f1_mat[inda, indr] = metrics(TP_fnA, TN_fnA, FP_fnA, FN_fnA)[2]

            rho += up
            if indr is not None:
                indr += 1

        alpha += up
        inda += 1
        rho = 0.1

        if adaptive:
            indr = 0
        else:
            tot = TP_fnA + TN_fnA + FP_fnA + FN_fnA
            TestATP.append(TP_fnA / float(tot))
            TestATN.append(TN_fnA / float(tot))
            TestAFP.append(FP_fnA / float(tot))
            TestAFN.append(FN_fnA / float(tot))
            prec.append(metrics(TP_fnA, TN_fnA, FP_fnA, FN_fnA)[1])
            recall.append(metrics(TP_fnA, TN_fnA, FP_fnA, FN_fnA)[0])
            rho = 0

    f1_coord = np.unravel_index(np.argmax(f1_mat), f1_mat.shape)
    pickle.dump(f1_mat,open(dataset+'_f1.p','wb'))

    alpha = up * (f1_coord[0] + 1)

    if adaptive:
        rho = up * (f1_coord[1] + 1)

        plt.figure(figsize=(10, 10))
        plt.title('Alpha vs Rho vs f1 (adaptive) - Dataset: ' + dataset)
        plt.imshow(f1_mat, extent=[rho_start, maxr, alpha_start, maxa])
        plt.scatter(rho, alpha)
        plt.xlabel('Rho')
        plt.ylabel('Alpha')
        axes = plt.gca()
        axes.set_xlim([rho_start, maxr])
        axes.set_ylim([alpha_start, maxa])

        if show_plt:
            plt.show()
            plt.close()
        plt.savefig('F1_2ad_' + dataset + '.png')

    else:
        title = dataset + ' DS frms ' + str(te_frmStart) + '-' + str(te_frmEnd) + ' - $\\alpha$ [0.1-10.0] '
        roc(recall, prec, title, show_plt, dataset)

        alphalst = np.arange(alpha_start, maxa+up, up)
        fig, ax = plt.subplots()
        ax.plot(alphalst, TestATP, c='g', label='TP')
        ax.plot(alphalst,TestAFN, c='b', label='FN')
        ax.plot(alphalst,TestAFP, c='r', label='FP')
        ax.plot(alphalst,TestATN, c='black', label='TN')
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        plt.axis([0, maxa, 0, 1])
        ax.xaxis.set_ticks(np.arange(0, maxa, 1))
        ax.yaxis.set_ticks(np.arange(0, 1.01, 0.1))
        plt.title('Alpha vs f1 (non-adaptive) - Dataset: ' + dataset)
        plt.plot(alphalst, f1, c='r', label='F1')
        plt.plot(alphalst, recall, c='b', label='Recall')
        plt.plot(alphalst, prec, c='g', label='Precision')
        plt.legend(loc='lower right')
        plt.xlabel('$\\alpha$')
        plt.ylabel('Metrics')
        annot_max(np.asarray(alphalst), np.asarray(f1))

        if show_plt:
            plt.show()
            plt.close()

        plt.savefig('F1_2_' + dataset + '.png')

    return alpha, rho


def roc(recall_lst, precision_lst, title='', show_plt=False, dataset='highway'):
    recall = np.asarray(recall_lst)
    precision = np.asarray(precision_lst)
    precision2 = precision.copy()
    i = recall.shape[0] - 2

    # interpolation...
    while i >= 0:
        if precision[i + 1] > precision[i]:
            precision[i] = precision[i + 1]
        i = i - 1

    # plotting...
    fig, ax = plt.subplots()
    ax.plot(recall, precision2, 'k--', color='blue')
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")

    precision, recall = precision_lst, recall_lst
    skmetrics.auc(precision, recall, reorder=True)
    plt.step(recall, precision, color='b', alpha=0.1,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.1,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([min(recall_lst), 1.0])
    print skmetrics.auc(recall, precision, reorder=True)
    plt.title(title + 'AUC={0:0.2f}'.format(skmetrics.auc(recall, precision, reorder=True)))

    if show_plt:
        fig.show()

    plt.savefig('ROC_' + dataset + '.png')


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


def createMOG(hist=150, thr=10, shadows=False):
    return cv2.createBackgroundSubtractorMOG2(history=hist, varThreshold=thr, detectShadows=shadows)
