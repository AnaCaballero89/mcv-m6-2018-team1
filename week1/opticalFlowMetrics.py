import math
import numpy as np
import matplotlib.pyplot as plt

# Compute MSEN and PEPN for an estimated Optical Flow image using ground truth
def opticalFlowMetrics(flowResult, flowGT, frameName=0, plot=True):

    # To compute MSEN (Mean Square Error in Non-occluded areas)
    distances = []
    # Percentage of Erroneous Pixels in Non-occluded areas (PEPN)
    errPixels = []
    # Euclidean distance threshold for PEPN
    errorTh = 3
    errorImage = []

    # Loop through each pixel
    for i in range(np.shape(flowResult)[0]):
        for j in range(np.shape(flowResult)[1]):
            # Convert u-/v-flow into floating point values
            convertedPixelResult_u = (float(flowResult[i][j][1])-2**15)/64.0
            convertedPixelResult_v = (float(flowResult[i][j][2])-2**15)/64.0
            convertedPixelGT_u = (float(flowGT[i][j][1])-2**15)/64.0
            convertedPixelGT_v = (float(flowGT[i][j][2])-2**15)/64.0
            # If ground truth is available, compare it to the estimation result using Euclidean distance
            if flowGT[i][j][0] == 1:
                dist = math.sqrt((convertedPixelResult_u-convertedPixelGT_u)**2+(convertedPixelResult_v-convertedPixelGT_v)**2)
                distances.append(dist)
                errorImage.append(dist)
                # If the distance is more than the threshold, consider the pixel as erroneous
                if abs(dist) > errorTh:
                    errPixels.append(True)
                else:
                    errPixels.append(False)
            else:
                errorImage.append(0)

    msen = np.mean(distances)
    pepn = np.mean(errPixels)*100

    # Print[, plot] and return results
    if frameName != 0:
        print "\n###########\nFrame: ", frameName

    print "Mean Square Error in Non-occluded areas (MSEN): ", msen
    print "Percentage of Erroneous Pixels in Non-occluded areas (PEPN): ", pepn, "%"

    cm = plt.cm.get_cmap('RdYlBu_r')

    # Get the histogram
    Y,X = np.histogram(distances, 20, normed=True)
    x_span = X.max()-X.min()
    C = [cm(((x-X.min())/x_span)) for x in X]

    plt.bar(X[:-1],Y*100,color=C,width=X[1]-X[0])

    plt.xlabel('Distance error')
    plt.ylabel('N. of pixels (%)')
    plt.title('Histogram of Error per pixel')
    plt.show()

    return msen, pepn, errorImage
