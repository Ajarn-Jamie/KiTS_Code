from starter_code.utils import load_case
import starter_code.myKitsCode as mkc
import matplotlib.pyplot as plt
from time import time
import pickle
import numpy as np

def poly_optimize(erp, times,deg):
    equ = np.polyfit(times, erp, deg=deg)
    equ = np.poly1d(equ)
    poly_line = equ(times)
    return poly_line


##cases = [0, 5, 8, 9, 26, 29, 31, 35, 46, 47, 51, 68, 79, 80, 86, 88,
##         94, 102, 103, 105, 124, 136, 138, 146, 153, 162,
##         168, 170, 175, 176, 178, 186, 189, 190, 201, 202, 203, 209]

cases = [0]
for i in cases:
##for i in range(210):
    start_time = time()
    print('case_00%.3i'%i)
    # Load processed data
    with open('D:\DATA_KITS2019\kits19\prep_data\case_00%.3i.pickle'%i, 'rb') as f:
        x_prep, y_prep, restore_dat = pickle.load(f)

    # Prepare data for plotting
    rest = np.array(restore_dat)
    x_axis = np.linspace(0,x_prep.shape[0],x_prep.shape[0])

    # Fit polynomials
    x_adj_pol = poly_optimize(rest[:,0], x_axis,4)
    y_adj_pol = poly_optimize(rest[:,1], x_axis,4)
    w_pol = poly_optimize(rest[:,2], x_axis,4)
    h_pol = poly_optimize(rest[:,3], x_axis,4)

    # Plot restore data
    plt.subplot(121)
    plt.plot(x_axis,rest[:,0],label='X Adj.')
    plt.plot(x_axis,rest[:,1],label='Y Adj.')
##    plt.plot(x_axis,x_adj_pol,label='X Poly')
##    plt.plot(x_axis,y_adj_pol,label='Y Poly')
##    plt.axis([0, 128, 0, 128])
    plt.legend()
    plt.subplot(122)
    plt.plot(x_axis,rest[:,2],label='W')
    plt.plot(x_axis,rest[:,3],label='H')
##    plt.plot(x_axis,w_pol,label='W Poly')
##    plt.plot(x_axis,h_pol,label='H Poly')
    plt.ylim([0, 512])
    plt.legend()
    plt.suptitle('Case 00%.3i: 4th Poly Fits'%i)

##    plt.savefig('D:\DATA_KITS2019\plots_14\case_00%.3i'%i,dpi=200)
##    plt.clf()

    print('Processing time = ', time()-start_time)

plt.show()
##    img_rest, seg_rest = mkc.restore_images(img_prep,seg_prep,rest)   
##    seg_diff = seg_orig - seg_rest

