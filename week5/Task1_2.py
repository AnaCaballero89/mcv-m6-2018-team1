from utils import *
import sys
from scipy import ndimage
from math import sqrt
import time


def appendTracker(trackers, tracker_type):
    (_, minor_ver, _) = (cv2.__version__).split('.')

    if int(minor_ver) < 3:
        trackers.append(cv2.Tracker_create(tracker_type))
    else:
        if tracker_type == 'KCF':
            trackers.append(cv2.TrackerKCF_create())
        if tracker_type == 'MEDIANFLOW':
            trackers.append(cv2.TrackerMedianFlow_create())


def task1_2(dataset,tr_frmStart,tr_frmEnd):

    # Reset tmp folders
    if 'tmpSequence' in os.listdir('../datasets/'+dataset):
        shutil.rmtree('../datasets/'+dataset+'/tmpSequence/')
    os.makedirs('../datasets/'+dataset+'/tmpSequence')

    if 'task1_2Results' in os.listdir("."):
        shutil.rmtree('task1_2Results')
    os.makedirs('task1_2Results/'+dataset)

    trackers = []               # List of trackers
    areaThreshold = 150         # Threshold to avoid noise in the foreground
    distThreshold = 20          # Threshold to track previously detected objects

    tracker_types = ['KCF', 'MEDIANFLOW']
    tracker_type = tracker_types[1]

    # Read video
    cap = cv2.VideoCapture(dataset+'FG/'+dataset+'.mp4')

    # Exit if video not opened.
    if not cap.isOpened():
        print "Could not open video"
        sys.exit()

    # Read first frame.
    ok, frame = cap.read()

    centroidList1 = []
    centroidList2 = []
    firstFlag = False
    topMargin = 0.95
    bottomMargin = 0.05

    frameId = 0
    createTmpSequence(tr_frmStart, tr_frmEnd, dataset)
    cap2 = cv2.VideoCapture('../datasets/' + dataset + '/tmpSequence/in%03d.jpg')

    while cap.isOpened():
        bboxList1 = []
        bboxList2 = []

        frame = ndimage.gaussian_filter(frame, 1.0)
        frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)[1]
        frame = frame[:,:,0]

        output = cv2.connectedComponentsWithStats(frame)

        # output = output[1:]
        stats = output[2][1:]

        _, rgbMask = cap2.read()

        if len(stats>2):
            centroids = output[3][1:]
            for i, centroid in enumerate(centroids):
                # If the area of the connected component is not big enough, ignore it
                if stats[i][4] > areaThreshold:
                    centroid = (map(int, centroid))
                    topLeftCorner = (stats[i][0], stats[i][1])
                    bottomRightCorner = (stats[i][0]+stats[i][2], stats[i][1]+stats[i][3])
                    bbox = (topLeftCorner[0] - 10, topLeftCorner[1] - 10, bottomRightCorner[0] - topLeftCorner[0] + 20,
                            bottomRightCorner[1] - topLeftCorner[1] + 20)

                    if (len(centroidList1) == 0 or firstFlag == True) and (centroid[1]) < np.shape(frame)[0]*topMargin and (centroid[1]) > np.shape(frame)[0]*bottomMargin:
                        centroidList1.append(centroid)
                        bboxList1.append(bbox)
                        firstFlag = True
                    elif (centroid[1]) < np.shape(frame)[0]*topMargin and (centroid[1]) > np.shape(frame)[0]*bottomMargin:
                        centroidList2.append(centroid)
                        bboxList2.append(bbox)

            firstFlag = False

            trackerCounter = 0
            trackersToBeRemoved = []
            newBboxes = []

            # For each centroid, check if it's a new detection by calculating the distance with the old centroids
            if len(centroidList2) > 0:
                newDetectFlag = [True]*len(centroidList2)
                for i, c2 in enumerate(centroidList2):

                    for c1 in centroidList1:

                        dist = sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

                        # If it's less than the threshold, track it
                        if dist < distThreshold:

                            newDetectFlag[i] = False
                            ok, bbox = trackers[trackerCounter].update(frame)

                            # Draw bounding box or set tracker to be deleted
                            if (bbox[1]+bbox[3]) > np.shape(frame)[0] or not ok:
                                trackersToBeRemoved.append(trackerCounter)
                            else:
                                # Tracking success
                                p1 = (int(bbox[0]), int(bbox[1]))
                                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                                cv2.rectangle(rgbMask, p1, p2, (255, 0, 0), 2, 1)
                                trackerCounter += 1
                            break

                    # If a new detection, start a new tracker
                    if newDetectFlag[i] == True:
                        topLeftCorner = (bboxList2[i][0], bboxList2[i][1])
                        bottomRightCorner = (bboxList2[i][0] + bboxList2[i][2], bboxList2[i][1] + bboxList2[i][3])
                        bbox = (topLeftCorner[0] - 10, topLeftCorner[1] - 10, bottomRightCorner[0] - topLeftCorner[0] + 20,
                                bottomRightCorner[1] - topLeftCorner[1] + 20)
                        newBboxes.append(bbox)

                        appendTracker(trackers, tracker_type)
                        ok = trackers[-1].init(frame, bbox)
                        cv2.rectangle(rgbMask, topLeftCorner, bottomRightCorner, (255, 0, 0), 2, 1)
                        print "[2] I just created a tracker (N.:", len(trackers), ")"


                    if (c2[1]) > np.shape(frame)[0]*topMargin or (c2[1] < np.shape(frame)[0]*bottomMargin):
                        trackersToBeRemoved.append(trackerCounter)
                        newDetectFlag[i] = False

            else:
                for i in range(len(centroidList1)):
                    appendTracker(trackers, tracker_type)
                    topLeftCorner = (bboxList1[i][0], bboxList1[i][1])
                    bottomRightCorner = (bboxList1[i][0] + bboxList1[i][2], bboxList1[i][1] + bboxList1[i][3])
                    bbox = (topLeftCorner[0] - 10, topLeftCorner[1] - 10, bottomRightCorner[0] - topLeftCorner[0] + 20,
                            bottomRightCorner[1] - topLeftCorner[1] + 20)
                    ok = trackers[i].init(frame, bbox)
                    print "[1] I just created a tracker (N.:", len(trackers),")"
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(rgbMask, p1, p2, (255, 0, 0), 2, 1)

            for c in trackersToBeRemoved:
                trackers[c] = "DELETE_THIS_TRACKER"
                centroidList2[c] = "DELETE_THIS_CENTROID"

            trackers = [t for t in trackers if t != "DELETE_THIS_TRACKER"]
            centroidList2 = [c for c in centroidList2 if c != "DELETE_THIS_CENTROID"]
            # print "how many trackers: ", len(trackers)

        cv2.imshow('',rgbMask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Save frame
        cv2.imwrite('task1_2Results/' + dataset + '/' + str(frameId) + '.png', rgbMask)
        frameId += 1
        ok, frame = cap.read()
        if not ok:
            break

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break

        # Update the two lists of centroids
        if len(centroidList2) > 0:
            centroidList1 = centroidList2
            centroidList2 = []

    print "Final number of trackers: ", len(trackers)