##https://www.programcreek.com/python/example/88831/skimage.measure.regionprops
import pickle, os#, xlrd
import numpy as np
import matplotlib.pyplot as plt

##f1_scores = np.load('f1_scores_vvKT2.npy')
##cases = np.nonzero(f1_scores[:,2]<0.1)

##cases = [15,19,20,37,83,103,120,121,137,164,165,182,188,191,194,198,208]
cases = [0,56,103]
for i in cases:
##for i in range(210):
    print('case_00%.3i'%i)
    with open('D:\DATA_KITS2019\kits19\prep_data2\case_00%.3i.pickle'%i, 'rb') as f:
        x, y_true, restore_dat = pickle.load(f)
    with open('D:\DATA_KITS2019\PredictionsVV\case_00%.3i.pickle'%i, 'rb') as f:
        x, y_pred, restore_dat = pickle.load(f)

    
    for j in range(0,x.shape[0],5):
        path = 'D:\DATA_KITS2019\plots_18\case_00%.3i'%i
        if not os.path.exists(path):
            os.makedirs(path)
        imfilename = 'image_%.6i.png' % j
        impath = os.path.join(path, imfilename)
        for k in range(2):
            plt.subplot(1,2,k+1)
            plt.imshow(x[j],'gray'),plt.axis('off')
            if k == 0:
                plt.imshow(y_true[j],'jet',alpha=0.5,vmin=0,vmax=2),plt.axis('off')
                plt.title('Ground Truth')
            if k == 1:
                plt.imshow(y_pred[j],'jet',alpha=0.5,vmin=0,vmax=2),plt.axis('off')
                plt.title('Prediction')
        plt.savefig(impath,dpi=200)
        print('Frame %i Complete'%j)
