from starter_code.utils import load_case
import starter_code.myKitsCode as mkc
import matplotlib.pyplot as plt
from time import time
import pickle
import os

##cases = [0]
##for i in cases:
for i in range(1,210):
    start_time = time()
    print('case_00%.3i'%i)
    # Load processed data
    with open('D:\DATA_KITS2019\kits19\prep_data\case_00%.3i.pickle'%i, 'rb') as f:
        x_prep, y_prep, restore_dat = pickle.load(f)
        
    # Create directory to save frame images
    im_dir = r'D:\DATA_KITS2019\plots_11\case_00%.3i' % i
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)

    # Plot and save every 5th frame
    for j in range(0,x_prep.shape[0],5):
        print('Slice ', j)
        plt.imshow(x_prep[j],'gray'),plt.axis('off')
        plt.title('Case_00%.3i Slice %i'%(i,j))
        plt.imshow(y_prep[j],'jet',vmax=2,alpha=0.3)
        imfilename = 'case_00%.3i_slice%.3i.png' % (i,j)
        impath = os.path.join(im_dir, imfilename)
        plt.savefig(impath,dpi=200)
        plt.clf()

    print('Processing time = ', time()-start_time)


##    img_rest, seg_rest = mkc.restore_images(img_prep,seg_prep,rest)   
##    seg_diff = seg_orig - seg_rest

##import matplotlib.pyplot as plt
##plt.imshow(x_prep[100],'gray'),plt.imshow(y_prep[100],'jet',alpha=0.3),plt.show()
