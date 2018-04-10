import cv2
import numpy
import os

def task2_2():
    matstack = numpy.identity(3)
    matlog = []
    p = '../datasets/traffic/'
    inp = ['input/' + x for x in os.listdir(p + 'input/')]
    inp.sort()
    gth = ['groundtruth/' + x for x in os.listdir(p + 'groundtruth/')]
    gth.sort()
    phv = '../datasets/highway/input/'
    # gth=[x for x in os.listdir(phv)]
    old = cv2.cvtColor(cv2.imread(p + inp[0]), cv2.COLOR_BGR2GRAY)
    for im in inp:
        img = cv2.imread(p + im)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old, 400, 0.01, 10)

        cur_features, st, err = cv2.calcOpticalFlowPyrLK(old, gray, old_features, None)

        st[sum(((old_features - cur_features) ** 2).reshape((len(st), 2)).transpose((1, 0))) > 64 ** 2] = 0
        for old, cur in zip(old_features[st == 1], cur_features[st == 1]):
            old = tuple(old)
            cur = tuple(cur)

            cv2.line(img, old, cur, (0, 255, 0), 1)
            cv2.circle(img, cur, 4, (0, 255, 0), 1)

        if st.sum() == 0:
            matlog.append(matstack[:2])
        else:
            mat = cv2.estimateRigidTransform(old_features[st == 1], cur_features[st == 1], False)

            if mat is None:
                matlog.append(matstack[:2])
            else:
                mat = numpy.append(mat, numpy.array([[0, 0, 1]]), axis=0)
                matstack = matstack.dot(mat)
                matlog.append(matstack[:2])

        #print(','.join(str(x) for x in matstack[:2].reshape(6)))

        # cv2.imshow('frame', img)
        # cv2.waitKey(1)

        old = gray

    #
    # cam = cv2.VideoCapture('in.mp4')
    # fps = cam.get(cv2.CAP_PROP_FPS)
    fps = 50
    # out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'X264'), 50, (
    #    int(320),
    #    int(270),
    # ))
    # out.write(cam.read()[1])

    for i, mat in enumerate(matlog):
        img = cv2.imread(p + inp[i])
        gtimg = cv2.cvtColor(cv2.imread(p + gth[i]), cv2.COLOR_BGR2GRAY)

        d = matlog[max(0, i - int(fps)):min(i + int(fps), len(matlog) - 1)]
        avg = numpy.append(sum(d) / len(d), numpy.array([[0, 0, 1]]), axis=0)
        mat = numpy.append(mat, numpy.array([[0, 0, 1]]), axis=0)
        dot = numpy.linalg.inv(mat).dot(avg)

        warp = cv2.warpAffine(img, dot[:2], img.shape[1::-1])
        warp2 = cv2.warpAffine(gtimg, dot[:2], gtimg.shape[1::-1])
        cv2.imwrite('week4Results/Results' + str(i) + '.jpg', warp)
        cv2.imwrite('week4Results/Resulst/gt/' + str(i) + '.jpg', warp2)
        # cv2.imshow('smoothing', warp)

        #    out.write(warp)
        cv2.waitKey(1)

