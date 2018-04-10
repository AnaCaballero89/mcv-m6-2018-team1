from utils import *
from Task1_1 import task1_1
from Task1_2 import task1_2
from Task2 import task2
from Task3 import task3


def main():
    dataset = 'highway'  # 'highway', 'traffic'

    inputpath = '../datasets/' + dataset + '/input/'
    groundTruthPath = '../datasets/' + dataset + '/groundtruth/'

    if dataset == 'highway':
        tr_frmStart = 1050
        tr_frmEnd = 1350
        te_frmStart = 1
        te_frmEnd = 1700
        MOGthreshold = 10

    elif dataset == 'traffic':
        tr_frmStart = 950
        tr_frmEnd = 1050
        te_frmStart = 1
        te_frmEnd = 1570
        MOGthreshold = 330
    else:
        print "You haven't defined the right dataset. Options are: highway or traffic."
        exit(0)

    # Tracking with Kalman Filters
    task1_1(MOGthreshold, inputpath, dataset)

    # Tracking with other method
    task1_2()

    # Speed Estimator
    task2()

    # Software
    task3()


main()