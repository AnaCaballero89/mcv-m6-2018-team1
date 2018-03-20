from utils import *

def task1_1(inputpath, groundTruthPath, dataset, ret=False):

    window_size = np.array([50])  # np.array([10, 20, 30, 40, 50])
    look_area = np.array([2, 4])  # np.array([10, 20, 30, 40, 50])
    step = 10
    images = read_of_images(inputpath)
    groundtruth = readOF(groundTruthPath)

    all_msen = np.zeros((len(window_size), len(look_area), len(images) - 1))
    all_pepn = np.zeros((len(window_size), len(look_area), len(images) - 1))
    vector_field_ret = []

    for wi in range(0, window_size.shape[0]):
        for li in range(0, look_area.shape[0]):
            for i in range(1, len(images)):
                print 'Getting Optical Flow'
                vector_field = block_matching(images[i-1], images[i], window_size[wi], look_area[li], step)
                vec_field_mat = np.ones((vector_field[0].shape[0], vector_field[0].shape[1], 3))
                vec_field_mat[:, :, 1] = vector_field[0]
                vec_field_mat[:, :, 2] = vector_field[1]

                gt = groundtruth[i-1][range(0, int((images[i-1].shape[0] - window_size[wi]) / step) * step + step, step), :, :] \
                                [:, range(0, int((images[i-1].shape[1] - window_size[wi]) / step) * step + step, step), :]

                print 'Gettin measures'
                msen, pepn, _ = opticalFlowMetrics(vec_field_mat, gt, normInp=False)
                all_msen[wi, li, i - 1] = msen
                all_pepn[wi, li, i - 1] = pepn

    for i in range(0, len(window_size)):
        plt.plot(look_area, all_msen[i, :, :], label='Win Size ' + str(window_size[i]))
    plt.xlabel('Area of search')
    plt.ylabel('MSEN%')
    plt.legend()
    plt.title('Seq 157')
    plt.show()
    plt.close()

    for i in range(0, len(window_size)):
        plt.plot(look_area, all_pepn[i, :, :], label='Win Size ' + str(window_size[i]))
    plt.xlabel('Area of search')
    plt.ylabel('PEPN%')
    plt.legend()
    plt.title('Seq 157')
    plt.show()
    plt.close()
