
from utils import *
def task1_2(indir, gtdir):
    fls=[x for x in os.listdir(indir) if x[-4:] in ['.png','.jpg']]
    fls.sort()
    i=1
    while i<len(fls):
        frame1 = cv2.imread(indir+fls[i-1])
        frame2 = cv2.imread(indir+fls[i])
        gt=np.squeeze(readOF(gtdir))
        flowResult=OF_Farneback(frame1, frame2, outdir=None, fn='0', winsize=10)
        vec_field_mat = np.ones((gt.shape[0], gt.shape[1], 3))
        
        vec_field_mat[:, :, 1] = flowResult[:,:,0]
        vec_field_mat[:, :, 2] = flowResult[:,:,1]
        opticalFlowMetrics(vec_field_mat, gt, normInp=False)

        i+=1
