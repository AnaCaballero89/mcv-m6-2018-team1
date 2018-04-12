from utils import *
import cv2
import copy
from detectors import Detectors
from tracker import Tracker



def task1_1(mogthr, inputpath, dataset):
    # Create opencv video capture object
    path = inputpath + 'in%06d.jpg'
    cap = cv2.VideoCapture(path)

    # Create Object Detector
    detector = Detectors(thr=mogthr, dataset=dataset)

    # Create Object Tracker
    if dataset == 'highway':
        tracker = Tracker(200, 0, 60, 100)  # Tracker(200, 0, 200, 100)
    elif dataset == 'traffic':
        tracker = Tracker(200, 0, 60, 100)  # Tracker(50, 0, 90, 100)
    elif dataset == 'ownhighway':
        tracker = Tracker(45, 0, 60, 100)  # Tracker(50, 0, 90, 100)
    # Variables initialization
    skip_frame_count = 0
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    pause = False

    if dataset == 'highway':
        pts1 = np.float32([[120, 100], [257, 100], [25, 200], [250, 200]])
    elif dataset == 'traffic':
        pts1 = np.float32([[0, 50], [160, 15], [110, 190], [320, 110]])
    elif dataset == 'ownhighway':
        pts1 = np.float32([[190, 100], [290, 100], [60, 200], [250, 200]])
    pts2 = np.float32([[0, 0], [320, 0], [0, 240], [320, 240]])

    M = cv2.getPerspectiveTransform(pts1, pts2)


    print M
    counter = 0
    # Infinite loop to process video frames
    while True:
        counter += 1
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Stop when no frame
        if frame is None:
            break

        # Make copy of original frame
        orig_frame = copy.copy(frame)

        # Skip initial frames that display logo
        #if (skip_frame_count < 200):
        #    skip_frame_count += 1
        #    continue

        # Detect and return centeroids of the objects in the frame
        centers, xd, yd, wd, hd = detector.Detect(frame, counter)
        #print xd

        vel = []
        # If centroids are detected then track them
        if (len(centers) > 0):

            # Track object using Kalman Filter
            tracker.Update(centers, dataset)



            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > 1):
                    vel = []
                    a=0
                    for j in range(5, len(tracker.tracks[i].trace) - 1):
                        a=a+1
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j + 1][0][0]
                        y2 = tracker.tracks[i].trace[j + 1][1][0]
                        clr = tracker.tracks[i].track_id % 9
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), track_colors[clr], 2)
                        if dataset == 'highway':
                            if y1 > 100 and y2 < 200:
                                x1r, y1r, z1r = np.dot(M,[x1, y1, 1])
                                x2r, y2r, z2r = np.dot(M, [x2, y2, 1])
                                x1r, y1r = x1r/z1r, y1r/z1r
                                x2r, y2r = x2r / z2r, y2r / z2r
                                dist = np.float(
                                    np.sqrt(((int(x2r) - int(x1r)) ** 2) + ((int(y2r) - int(y1r)) ** 2))) * np.float(
                                    30) / 20 * np.float(24) / 5  # euclidean distance between two points
                                vel.append(dist)
                        if dataset == 'ownhighway':
                            if y1 > 100 and y2 < 200:
                                x1r, y1r, z1r = np.dot(M,[x1, y1, 1])
                                x2r, y2r, z2r = np.dot(M, [x2, y2, 1])
                                x1r, y1r = x1r/z1r, y1r/z1r
                                x2r, y2r = x2r / z2r, y2r / z2r


                                dist = np.float(
                                    np.sqrt(((int(x2r) - int(x1r)) ** 2) + ((int(y2r) - int(y1r)) ** 2))) * np.float(
                                    18) / 20 * np.float(24) / 5  # euclidean distance between two points
                                vel.append(dist)

                    if not vel == []:
                        #if i==1:
                        #print xd[i]#'value ' + xd[i] + ' value ' + yd[i]#+ ' frame '+frame+ ' vel ' +vel
                        # if dataset == 'ownhighway':
                        #     #if i==0:
                        #     print counter,i, xd,np.mean(vel)
                        #     if counter>0:
                        #         a=0
                        #
                        #
                        #     #if xd==[]
                        #     # if len(vel)<4: #and int(np.mean(vel))>100:
                        #     #     cv2.putText(frame, '  vel ' + str(int(np.mean(vel))), (int(xd[a]), int(yd[a] - 4)),
                        #     #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                        #     if len(vel)>3:# and int(np.mean(vel))>100:
                        #         cv2.putText(frame, '  vel ' + str(int(np.mean(vel[-3:-1]))), (int(xd[0]), int(yd[0])),
                        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                        #     #cv2.putText(frame, '  vel ' + str(int(np.mean(vel))), (int(xd[0]), int(yd[0] - 4)),
                        #     #    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                        #     #print int(np.mean(vel)),i,j
                        if dataset == 'ownhighway':
                            #print i, xd
                            if len(vel)<10:
                                cv2.putText(frame, '  vel ' + str(int(np.mean(vel))), (int(xd[0]), int(yd[0] - 4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                            else:
                                cv2.putText(frame, '  vel ' + str(int(np.mean(vel[-10:-1]))), (int(xd[0]), int(yd[0] - 4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                        if dataset == 'highway':
                            #print i, xd
                            if len(vel)<20:
                                cv2.putText(frame, '  vel ' + str(int(np.mean(vel))), (int(xd[i]), int(yd[i] - 4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                            else:
                                cv2.putText(frame, '  vel ' + str(int(np.mean(vel[-20:-1]))), (int(xd[i]), int(yd[i] - 4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                            #print int(np.mean(vel)), i, j
                    # x1 = tracker.tracks[i].trace[-2][0][0]
                    # y1 = tracker.tracks[i].trace[-2][1][0]
                    # x2 = tracker.tracks[i].trace[-1][0][0]
                    # y2 = tracker.tracks[i].trace[-1][1][0]
                    # if dataset == 'highway':
                    #     if y1 > 100 and y2 < 200:
                    #         x1r, y1r, z1r = np.dot(M,[x1, y1, 1])
                    #         x2r, y2r, z2r = np.dot(M, [x2, y2, 1])
                    #         x1r, y1r = x1r/z1r, y1r/z1r
                    #         x2r, y2r = x2r / z2r, y2r / z2r
                    #
                    #
                    #
                    #
                    #         dist = np.float(np.sqrt(((int(x2r) - int(x1r))**2) + ((int(y2r) - int(y1r))**2))) * np.float(30)/20 * np.float(24)/5#euclidean distance between two points
                    #         vel.append(dist)
                    #         cv2.putText(frame, '  vel '+str(int(dist)), (int(xd[i]), int(yd[i]-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),1,cv2.LINE_AA)
                                #print (x1, x2, y1, y2, dist, i)

                                # x1r,y1r = M * [x1,y1,1]            #(x,y,1) = M * (xold,yold,1)
                                # x2r, y2r = M * [x2,y2,1]
                                #
                                # vel.append[j] = int(np.sqrt(((int(x2r) - int(x1r)) ** 2) + (
                                #     (int(y2r) - int(y1r)) ** 2))) * int(3/50) * int(24/5) #     * (m/pixel) * (frame/sec)    # euclidean distance between two points
                                #
                                # if len(vel[j] > 10):
                                #     return#velocity = np.mean(vel[j](i:i-10))                 #     * (m/pixel) * (frame/sec)

                                #print 'car '+ str(i) +' velocity '+ str(dist) #(x pixels every frame) -> * (m/pixel) * (frame/sec) = (m/sec)















        # Display homography
        dst = cv2.warpPerspective(frame, M, (320, 240))
        cv2.imshow('Homography', dst)
        cv2.imwrite('../week5/results/hom' + str(counter) + '.png', dst)

        # Display the resulting tracking frame


        cv2.imshow('Tracking', frame)
        cv2.imwrite('../week5/results/out' + str(counter) + '.png', frame)
        cv2.imwrite('out' + str(frame) + '.jpg',frame)
        # Display the original frame
        cv2.imshow('Original', orig_frame)

        # Slower the FPS
        cv2.waitKey(1)

        # Check for key strokes
        k = cv2.waitKey(1) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break
        if k == 112:  # 'p' has been pressed. this will pause/resume the code.
            pause = not pause
            if (pause is True):
                print("Code is paused. Press 'p' to resume..")
                while (pause is True):
                    # stay in this loop until
                    key = cv2.waitKey(30) & 0xff
                    if key == 112:
                        pause = False
                        print("Resume code..!!")
                        break


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


