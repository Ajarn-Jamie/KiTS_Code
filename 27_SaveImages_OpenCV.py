import pickle, os#, xlrd
import numpy as np
import cv2

##f1_scores = np.load('f1_scores_vvKT_final.npy')
##cases = np.nonzero(f1_scores[:,2]<0.1)[0]
##cases = np.nonzero(f1_scores[:,2]>0.9)[0]

##cases = [15,19,20,37,83,103,120,121,137,164,165,182,188,191,194,198,208]
cases = [0,56,103]
for i in cases:
##for i in range(1,210):
    print('case_00%.3i'%i)
    with open('D:\DATA_KITS2019\kits19\prep_data2\case_00%.3i.pickle'%i, 'rb') as f:
        x, y_true, restore_dat = pickle.load(f)
    with open('D:\DATA_KITS2019\PredictionsVV\case_00%.3i.pickle'%i, 'rb') as f:
        x, y_pred, restore_dat = pickle.load(f)

    
    for j in range(x.shape[0]):
        path = 'D:\DATA_KITS2019\plots_18\case_00%.3i_select'%i
        if not os.path.exists(path):
            os.makedirs(path)
        imfilename = 'image_%.6i.jpg' % j
        impath = os.path.join(path, imfilename)

        im = np.expand_dims(x[j], 2)
        im = np.repeat(im, 3, axis=2)
        im = np.concatenate([im,im],axis=1)

        mask = np.concatenate([y_true[j],y_pred[j]],axis=1)
        mask = np.expand_dims(mask, 2)
        mask = np.repeat(mask, 3, axis=2)

        kid_mask = np.array(mask==1).astype('uint8') * 127
        tum_mask = np.array(mask==2).astype('uint8') * 127
        
        colors = {"red": [1,0.,0.], "blue": [0.,0.,1]}
        colored_kid_mask = np.multiply(kid_mask, colors["red"])
        colored_tum_mask = np.multiply(tum_mask, colors["blue"])
        img = np.array(im+colored_kid_mask+colored_tum_mask).astype('uint8')

##        cv2.imshow('Fig',img)
##        cv2.waitKey(500)

        cv2.imwrite(impath,img)
        print('Frame %i Complete'%j)
