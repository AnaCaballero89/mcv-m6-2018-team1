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
        tracker = Tracker(50, 0, 150, 100)  # Tracker(200, 0, 200, 100)
    elif dataset == 'traffic':
        tracker = Tracker(200, 0, 40, 90)  # Tracker(50, 0, 90, 100)

    # Variables initialization
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    pause = False

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
        centers = detector.Detect(frame, counter)

        # If centroids are detected then track them
        if (len(centers) > 0):

            # Track object using Kalman Filter
            tracker.Update(centers, dataset)

            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > 1):
                    for j in range(5, len(tracker.tracks[i].trace) - 1):
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j + 1][0][0]
                        y2 = tracker.tracks[i].trace[j + 1][1][0]
                        clr = tracker.tracks[i].track_id % 9
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), track_colors[clr], 2)

        # Display the resulting tracking frame
        cv2.imshow('Tracking', frame)
        #cv2.imwrite('/Users/santiagoba88/Desktop/' + dataset + '/Tracking/' + str(counter) + '.png', frame)

        # Display the original frame
        #cv2.imshow('Original', orig_frame)

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


