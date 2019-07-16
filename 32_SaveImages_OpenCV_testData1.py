import pickle, os#, xlrd
import numpy as np
import cv2

##f1_scores = np.load('f1_scores_vvKT_final.npy')
##cases = np.nonzero(f1_scores[:,2]<0.1)[0]
##cases = np.nonzero(f1_scores[:,2]>0.9)[0]

##cases = [15,19,20,37,83,103,120,121,137,164,165,182,188,191,194,198,208]
##cases = [0,56,103]
##for i in cases:
for i in range(211,300):
    print('case_00%.3i'%i)
    with open('D:\DATA_KITS2019\kits19\prep_data_test\case_00%.3i.pickle'%i, 'rb') as f:
        x, y_true, restore_dat = pickle.load(f)

    
    for j in range(x.shape[0]):
        path = 'D:\DATA_KITS2019\plots_20\case_00%.3i_select'%i
        if not os.path.exists(path):
            os.makedirs(path)
        imfilename = 'image_%.6i.jpg' % j
        impath = os.path.join(path, imfilename)

        cv2.imwrite(impath,x[j])
        print('Frame %i Complete'%j)

    print('Case_%.6i COMPLETE'%i)
