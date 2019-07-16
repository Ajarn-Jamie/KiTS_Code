from starter_code.utils import load_volume
import starter_code.myKitsCode as mkc
from time import time
import cv2
import numpy as np
import pickle
import nibabel as nib
import os

for i in range(210):
    start_time = time()
    print('case_00%.3i'%i)
    # Load original data
    x_true = load_volume("case_00%.3i"%i)
    affine = x_true.affine
    x_true = x_true.get_data()      
    
    with open('D:\DATA_KITS2019\PredictionsVV\case_00%.3i.pickle'%i, 'rb') as f:
        x_prep, y_pred, restore_dat = pickle.load(f)

    y_restored = np.zeros(x_true.shape,'uint8')
    
    for j in range(x_true.shape[0]):      
##        _, y_restored[j] = mkc.restore_images(x_prep[j],y_pred[j],restore_dat[j])
        _, a = mkc.restore_images(x_prep[j],y_pred[j],restore_dat[j])
        y_restored[j] = cv2.resize(a,(y_restored[j].shape[1],y_restored[j].shape[0]))
        
        path = 'D:\DATA_KITS2019\plots_22\case_00%.3i'%i
        if not os.path.exists(path):
            os.makedirs(path)
        imfilename = 'image_%.6i.jpg' % j
        impath = os.path.join(path, imfilename)

        im = np.expand_dims(x_true[j], 2)
        im = np.repeat(im, 3, axis=2)

        mask = np.expand_dims(y_restored[j], 2)
        mask = np.repeat(mask, 3, axis=2)

        kid_mask = np.array(mask==1).astype('uint8') * 127
        tum_mask = np.array(mask==2).astype('uint8') * 127
        
        colors = {"red": [1,0.,0.], "blue": [0.,0.,1]}
        colored_kid_mask = np.multiply(kid_mask, colors["red"])
        colored_tum_mask = np.multiply(tum_mask, colors["blue"])
        img = np.array(im+colored_kid_mask+colored_tum_mask).astype('uint8')

        cv2.imwrite(impath,img)
        print('Case_%5i - Frame %i Complete'%(i,j))

    data_path = 'D:\DATA_KITS2019\predictions'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    datfilename = 'prediction_%.5i.nii.gz' % i
    datpath = os.path.join(data_path, datfilename)


    nib_data = nib.Nifti1Image(y_restored, affine)
    nib.save(nib_data, datpath)
    
    print('Processing time = ', time()-start_time)


