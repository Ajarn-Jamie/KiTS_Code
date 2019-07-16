from starter_code.utils import load_case
import starter_code.myKitsCode as mkc
##import matplotlib.pyplot as plt
from time import time
import pickle
import os
import cv2
import numpy as np

##cases = [0]
##for i in cases:
k = 0
for i in range(210):
    start_time = time()
    print('case_00%.3i'%i)
    # Load processed data
    with open('D:\DATA_KITS2019\kits19\prep_data2\case_00%.3i.pickle'%i, 'rb') as f:
        x_prep, y_prep, restore_dat = pickle.load(f)
        
##    if i < 180:
##        im_dir = 'D:\DATA_KITS2019\data\\train'
##    else:
##        im_dir = 'D:\DATA_KITS2019\data\\test'
    
    # Plot and save every 5th frame
    for j in range(0,x_prep.shape[0],5):
        print('Slice ', j)

##        np.save(im_dir+'\\image_%.6i.npy'%k, x_prep[j])
##        np.save(im_dir+'\\label_%.6i.npy'%k, y_prep[j])
        
##        plt.imshow(x_prep[j],'gray'),plt.axis('off')
##        plt.title('Case_00%.3i Slice %i'%(i,j))
##        plt.imshow(y_prep[j],'jet',vmax=2,alpha=0.3)
######        imfilename = 'image_%.6i.jpg' % k
######        impath = os.path.join('D:\DATA_KITS2019\data\\images', imfilename)
########        plt.savefig(impath,dpi=200)
########        plt.clf()
######        segfilename = 'label_%.6i.jpg' % k
######        segpath = os.path.join('D:\DATA_KITS2019\data\\labels', segfilename)
######        
######        cv2.imwrite(impath,x_prep[j])
######        cv2.imwrite(segpath,y_prep[j]*127)
######        k += 1
        path = 'D:\DATA_KITS2019\plots_16\case_00%.3i'%i
        if not os.path.exists(path):
            os.makedirs(path)
        imfilename = 'image_%.6i.jpg' % j
        impath = os.path.join(path, imfilename)
        
        cv2.imwrite(impath,x_prep[j])
    print('Processing time = ', time()-start_time)

