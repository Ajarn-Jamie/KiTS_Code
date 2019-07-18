from starter_code.utils import load_case
import starter_code.myKitsCode as mkc
import numpy as np
from time import time
import pickle

cases = [0]
for i in cases:
##for i in range(210):
    start_time = time()

    print("case_00%.3i"%i)
    volume, segmentation = load_case("case_00%.3i"%i)

    x = volume.get_data().astype('float32')
    slices = x.shape[0]
    y = mkc.fill_segments(segmentation.get_data())

    x_prep = np.zeros((slices,512,512),'uint8')
    y_prep = np.zeros((slices,512,512),'uint8')
    restore_dat = []
    
    for j in range(slices):
        x_prep[j], y_prep[j], rest = mkc.prepare_images(x[j],y[j])
        restore_dat.append(rest)

    # Save processed data
    with open('D:\DATA_KITS2019\kits19\prep_data\case_00%.3i.pickle'%i, 'wb') as f:
        pickle.dump([x_prep,y_prep,restore_dat], f)

    print('Processing time = ', time()-start_time)


##    img_rest, seg_rest = mkc.restore_images(img_prep,seg_prep,rest)   
##    seg_diff = seg_orig - seg_rest

##import matplotlib.pyplot as plt
##plt.imshow(x_prep[100],'gray'),plt.imshow(y_prep[100],'jet',alpha=0.3),plt.show()
