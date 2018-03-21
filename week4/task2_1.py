from scipy import ndimage
from utils import *


def task2_1(inputpath,groundTruthPath, dataset, tr_frmStart=None, tr_frmEnd=None, te_frmStart=None, te_frmEnd=None):
    window_size = 50
    look_area = 2
    step = 10
    images = []
    color_images = []
    color_imagesgt = []

    if tr_frmStart == None and tr_frmEnd == None and te_frmStart == None and te_frmEnd == None:
        images = read_of_images(inputpath)
        color_imagesgt = readOF(groundTruthPath)
        color_images = readOF(inputpath)
    else:
        ImgNames = os.listdir(inputpath)
        ImgNames.sort()
        for idx, name in enumerate(ImgNames):
            if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg'):
                if int(name[-8:-4]) >= tr_frmStart and int(name[-8:-4]) <= te_frmEnd:
                    images.append(cv2.cvtColor(cv2.imread(inputpath + name), cv2.COLOR_BGR2GRAY))
                    color_images.append(cv2.imread(inputpath + name))
        color_imagesgt=readGT(groundTruthPath,tr_frmStart, te_frmEnd)


    shutil.rmtree('../datasets/' + dataset + '/tmpSequence/input/')
    os.makedirs('../datasets/' + dataset + '/tmpSequence/input/')

    shutil.rmtree('../datasets/' + dataset + '/tmpSequence/groundtruth/')
    os.makedirs('../datasets/' + dataset + '/tmpSequence/groundtruth/')

    prev_images = images[0]

    cv2.imwrite('../datasets/' + dataset + '/tmpSequence/input/' + str(0) + '.png', color_images[0])
    cv2.imwrite('../datasets/' + dataset + '/tmpSequence/groundtruth/' + str(0) + '.png', color_imagesgt[0])

    for i in range(1, len(images)):
        print 'Getting Optical Flow'
        vector_field = block_matching(prev_images, images[i], window_size, look_area, step)

        # Get displacement
        mag, ang = cv2.cartToPolar(vector_field[0], vector_field[1])
        mags, times = np.unique(mag, return_counts=True)
        magf = mags[times.argmax()]

        angs, times = np.unique(ang, return_counts=True)
        angf = angs[times.argmax()]

        x = magf*np.cos(angf)
        y = magf*np.sin(angf)

        # Get homography
        H = np.array([[1, 0, -x], [0, 1, -y]],dtype=np.float32)

        # Apply homography to gray scale image
        images[i] = cv2.warpAffine(images[i], H, dsize=images[i].shape)

        # Apply homography to color image and store
        cv2.imwrite('../datasets/' + dataset + '/tmpSequence/input/' + str(i) + '.png', (cv2.warpAffine(color_images[i], H, dsize=(color_images[i].shape[1],color_images[i].shape[0]))))
        cv2.imwrite('../datasets/' + dataset + '/tmpSequence/groundtruth/' + str(i) + '.png', (cv2.warpAffine(color_imagesgt[i-1], H, dsize=(color_imagesgt[i-1].shape[1],color_imagesgt[i-1].shape[0]))))

        # Update reference frame
        prev_images = images[i]

    inputpath = '../datasets/' + dataset + '/tmpSequence/input/'
    groundTruthPath = '../datasets/' + dataset + '/tmpSequence/groundtruth/'
    color_imagesgt = readOF(groundTruthPath)

    ImgNames = os.listdir(inputpath)
    ImgNames.sort()

    cnn = 4
    print 'segunda parte'

    if cnn == 4:
        str_elem = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    else:
        str_elem = np.ones((3, 3))

    if not (tr_frmStart == None and tr_frmEnd == None and te_frmStart == None and te_frmEnd == None):
        mog2BG = createMOG(hist=150, thr=330, shadows=False)
        bgad = []

        for idx, name in enumerate(ImgNames):
            if idx <= (te_frmEnd-tr_frmStart+1)/2:
                if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg'):
                    # Learning
                    out = mog2BG.apply(cv2.imread(inputpath + name), learningRate=0.01)

        te_ind = 0
        for idx, name in enumerate(ImgNames):
            if idx > (te_frmEnd - tr_frmStart + 1) / 2:
                if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg'):

                    # Testing
                    im = mog2BG.apply(cv2.imread(inputpath + name), learningRate=0.01)

                    # Apply Hole-Filling
                    im = ndimage.binary_fill_holes(im / 255, str_elem).astype(int)

                    bgad.append(arfilt(im, 4, 2000))

                    te_ind += 1

    TP, TN, FP, FN = added_evaluation(np.asarray(color_imagesgt)[51:], np.asarray(bgad), shadow=False)
    Recall, Pres, F1 = metrics(TP, TN, FP, FN)

    AUC_stb = skmetrics.auc(Recall, Pres, reorder=True)
    lbl_stb = 'Stabilised, AUC %.2f' % AUC_stb

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(Recall, Pres, color='r', label=lbl_stb)
    plt.legend()

    plt.show()


    return 0