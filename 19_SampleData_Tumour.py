from starter_code.utils import load_case
import starter_code.myKitsCode as mkc
from time import time
import pickle
import os
import numpy as np

##cases = [0]
##for i in cases:
for i in range(210):
    start_time = time()
    print('case_00%.3i'%i)
    # Load processed data
    with open('D:\DATA_KITS2019\kits19\prep_data2\case_00%.3i.pickle'%i, 'rb') as f:
        x_prep, y_prep, restore_dat = pickle.load(f)
    rest_dat = np.array(restore_dat)    

##    # Create directory to save frame images
##    im_dir = r'D:\DATA_KITS2019\kits19\sampled_data4\case_00%.3i' % i
##    if not os.path.exists(im_dir):
##        os.makedirs(im_dir)

    # Find slices containing tumour
    y_max = y_prep.max(axis=(1,2))
    tum_slices = y_max==2
    x_tum = x_prep[tum_slices]
    y_tum = y_prep[tum_slices]
    rest_dat_tum = rest_dat[tum_slices]

    # Slices containing only kidney
    kid_slices = y_max==1
    if any(kid_slices):
    ##    kid_slices = np.random.choice(kid_slices,int(x.shape[0]/4),replace=False)
        x_kid = x_prep[kid_slices]
        y_kid = y_prep[kid_slices]
        rest_dat_kids = rest_dat[kid_slices]    
        indexes = np.random.randint(0,x_kid.shape[0],int(x_tum.shape[0]/4))
        x_kid = x_kid[indexes]
        y_kid = y_kid[indexes]
        rest_dat_kids = rest_dat_kids[indexes]    
        
        # Augment tumor data
        x_tum_2 = np.flip(x_tum,axis=2)
        y_tum_2 = np.flip(y_tum,axis=2)

        x_sampled = np.concatenate([x_tum,x_tum_2,x_kid],axis=0)
        y_sampled = np.concatenate([y_tum,y_tum_2,y_kid],axis=0)
        rest_dat_sampled = np.concatenate([rest_dat_tum,rest_dat_tum,
                                           rest_dat_kids],axis=0)
     

    with open('D:\DATA_KITS2019\kits19\sampled_data4\case_00%.3i.pickle'%i, 'wb') as f:
        pickle.dump([x_sampled,y_sampled,rest_dat_sampled], f)
   
    print('Processing time = ', time()-start_time)
