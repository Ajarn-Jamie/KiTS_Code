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
##    im_dir = r'D:\DATA_KITS2019\kits19\sampled_data\case_00%.3i' % i
##    if not os.path.exists(im_dir):
##        os.makedirs(im_dir)

    # Find slices containing kideny and sample half the amount of non-kidney
    y_mean = np.mean(y_prep,axis=(1,2))
    kid_slices = y_mean.nonzero()[0]
    x_kids = x_prep[kid_slices]
    y_kids = y_prep[kid_slices]
    rest_dat_kids = rest_dat[kid_slices]
    pad_num = int(kid_slices.shape[0]/4)
    
##    # Save sampled data
##    with open('D:\DATA_KITS2019\kits19\sampled_data2\case_00%.3i.pickle'%i, 'wb') as f:
##        pickle.dump([x_kids,y_kids,rest_dat_kids], f)

    all_slices = np.linspace(0,y_prep.shape[0]-1,y_prep.shape[0],dtype='int')
    nonkid_slices = np.delete(all_slices,kid_slices)

    if nonkid_slices.shape[0] < pad_num:
        replace = True
    else:
        replace = False

    pad_slices = np.random.choice(nonkid_slices,pad_num,replace=replace)
    x_pad = x_prep[pad_slices]
    y_pad = y_prep[pad_slices]
    rest_dat_pad = rest_dat[pad_slices]

    x_sampled = np.concatenate([x_kids,x_pad],axis=0)
    y_sampled = np.concatenate([y_kids,y_pad],axis=0)
    rest_dat_sampled = np.concatenate([rest_dat_kids,rest_dat_pad],axis=0)
    
    # Save sampled data
    with open('D:\DATA_KITS2019\kits19\sampled_data\case_00%.3i.pickle'%i, 'wb') as f:
        pickle.dump([x_sampled,y_sampled,rest_dat_sampled], f)
   
    print('Processing time = ', time()-start_time)

##    img_rest, seg_rest = mkc.restore_images(img_prep,seg_prep,rest)   
##    seg_diff = seg_orig - seg_rest
