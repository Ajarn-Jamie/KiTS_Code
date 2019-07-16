from starter_code.utils import load_case
import starter_code.myKitsCode as mkc
from time import time
import os
import cv2
import numpy as np

##cases = [0]
##for i in cases:
for i in range(210):
    start_time = time()
    print('case_00%.3i'%i)
    # Load original data
    volume, segmentation = load_case("case_00%.3i"%i)

    x = volume.get_data().astype('float32')
    slices = x.shape[0]
    y = segmentation.get_data().astype('uint8')      
    
    # Plot and save every 5th frame
    for j in range(0,x.shape[0],5):
        print('Slice ', j)

        im = cv2.normalize(x[j],None,0,255,cv2.NORM_MINMAX).astype('uint8')
        
        path = 'D:\DATA_KITS2019\plots_0\case_00%.3i'%i
        if not os.path.exists(path):
            os.makedirs(path)
        imfilename = 'image_%.6i.jpg' % j
        impath = os.path.join(path, imfilename)
        
        cv2.imwrite(impath,im)

    print('Processing time = ', time()-start_time)


