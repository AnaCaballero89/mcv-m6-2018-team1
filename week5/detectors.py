'''
    File name         : detectors.py
    File Description  : Detect objects in video frame
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
import cv2
from skimage import measure
from scipy import ndimage

# set to 1 for pipeline images
debug = 0


class Detectors(object):
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """
    def __init__(self, thr, dataset):
        """Initialize variables used by Detectors class
        Args:
            None
        Return:
            None
        """
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=thr, detectShadows=False)
        self.choiceOfDataset = dataset


    def arfilt(self, frame, connect=4, area_thresh=1):
        frame = (frame > 0).astype(int)
        labeled = measure.label(frame, neighbors=connect)
        props = measure.regionprops(labeled)
        area = np.zeros((len(props) + 1, 1))
        for prop in props:
            area[prop.label] = prop.area
        #fil_img = (area[labeled, 0] > area_thresh).astype(int)
        fil_img = (np.logical_and(area[labeled, 0] > area_thresh, area[labeled, 0] < 12000))
        return fil_img

    def Detect(self, frame, counter):
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """



        # Convert BGR to GRAY
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if debug == 1:
            cv2.imshow('gray', gray)

        # Perform Background Subtraction
        fgmask = self.fgbg.apply(gray, learningRate=0.01)

        if self.choiceOfDataset == 'highway':
            structelement = np.ones((5, 5), np.uint8)
            fgmask = self.arfilt(fgmask, area_thresh=220)
            out = ndimage.binary_fill_holes(fgmask).astype(np.float32)
            out = cv2.erode(out, structelement, iterations=1)
            out = cv2.dilate(out, structelement, iterations=1)
            out = cv2.dilate(out, structelement, iterations=1)
            out = cv2.erode(out, structelement, iterations=1)
            out = ndimage.binary_fill_holes(out).astype(np.float32) * 255
        if self.choiceOfDataset == 'ownhighway':
            structelement = np.ones((3, 3), np.uint8)
            fgmask = self.arfilt(fgmask, area_thresh=220)
            out = ndimage.binary_fill_holes(fgmask).astype(np.float32)
            out = cv2.erode(out, structelement, iterations=1)
            out = cv2.dilate(out, structelement, iterations=1)
            out = cv2.dilate(out, structelement, iterations=1)
            out = cv2.erode(out, structelement, iterations=1)
            out = ndimage.binary_fill_holes(out).astype(np.float32) * 255

        elif self.choiceOfDataset == 'traffic':
            structelement2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                                       [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

            structelement3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                                       [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                                       [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                       [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                       [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                       [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

            fgmask = self.arfilt(fgmask, area_thresh=750)
            out = ndimage.binary_fill_holes(fgmask).astype(np.float32)
            out = cv2.erode(out, structelement2, iterations=1)
            out = cv2.dilate(out, structelement2, iterations=1)
            out = cv2.dilate(out, structelement3, iterations=1)
            out = cv2.erode(out, structelement3, iterations=1)
            out = ndimage.binary_fill_holes(out).astype(int) * 255

        fgmask = np.uint8(out)

        if debug == 0:
            cv2.imshow('bgsub', fgmask)

        # Detect edges
        edges = cv2.Canny(fgmask, 50, 190, 3)

        if debug == 1:
            cv2.imshow('Edges', edges)

        # Retain only edges within the threshold
        ret, thresh = cv2.threshold(edges, 127, 255, 0)

        # Find contours
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if debug == 0:
            cv2.imshow('thresh', thresh)

        centers = []  # vector of object centroids in a frame
        # we only care about centroids with size of bug in this example
        # recommended to be tunned based on expected object size for
        # improved performance
        xf = []
        yf = []
        wf = []
        hf = []

        blob_radius_thresh = 30
        # Find centroid for each valid contours
        for cnt in contours:
            try:
                # Calculate and draw rectangle
                x, y, w, h = cv2.boundingRect(cnt)
                if np.logical_and(np.logical_and(h > blob_radius_thresh, w > blob_radius_thresh), np.float(h) / w < 2):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    b = np.array([[x + w / 2], [y + h / 2]])
                    centers.append(np.round(b))
                    xf.append(x)
                    yf.append(y)
                    wf.append(wf)
                    hf.append(hf)

            except ZeroDivisionError:
                pass

        # show contours of tracking objects
        # cv2.imshow('Track Bugs', frame)

        return centers, xf, yf, wf, hf
