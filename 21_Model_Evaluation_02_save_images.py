from time import time
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

##cases = [0]
##for i in cases:
for i in range(1,210):
    start_time = time()
    print('case_00%.3i'%i)
    # Load processed data
    with open('D:\DATA_KITS2019\kits19\prep_data2\case_00%.3i.pickle'%i, 'rb') as f:
        x, y_true, restore_dat = pickle.load(f)
    rest_dat = np.array(restore_dat)
    # Load predictions
    file = 'D:\DATA_KITS2019\Predictions00\case_00%.3i.npy'%i
    y_pred = np.load(file).astype('uint8')
    # Plot frames with true and predicted annotations
    for j in range(0,x.shape[0]):
        path = 'D:\DATA_KITS2019\plots_17\case_00%.3i'%i
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
            
    print('Processing time = ', time()-start_time)
