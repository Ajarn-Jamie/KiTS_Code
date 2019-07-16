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

##cases = [203]
##for i in cases:
for i in range(210):
    start_time = time()
    print('case_00%.3i'%i)

    volume, segmentation = load_case("case_00%.3i"%i)

    x = volume.get_data().astype('float32')
    slices = x.shape[0]
    
    y = mkc.fill_segments(segmentation.get_data().astype('uint8'))
##    y = segmentation.get_data()
    
    cent_dat = np.zeros((slices,6))
    # Get centring data
    for j in range(slices):
        cent_dat[j] = mkc.get_centring_data(x[j])
    
    # Fit polynomials
    x_axis = np.linspace(0,slices,slices)
    x_adj_pol = poly_optimize(cent_dat[:,0], x_axis,4)
    y_adj_pol = poly_optimize(cent_dat[:,1], x_axis,4)
    x_pol = poly_optimize(cent_dat[:,2], x_axis,4)
    x_pol = np.clip(x_pol,0,512)
    y_pol = poly_optimize(cent_dat[:,3], x_axis,4)
    w_pol = poly_optimize(cent_dat[:,4], x_axis,4)
    h_pol = poly_optimize(cent_dat[:,5], x_axis,4)

    
##    # Plot restore data
##    plt.subplot(131)
##    plt.plot(x_axis,cent_dat[:,0],label='X Adj.')
##    plt.plot(x_axis,cent_dat[:,1],label='Y Adj.')
##    plt.plot(x_axis,x_adj_pol,label='X Poly')
##    plt.plot(x_axis,y_adj_pol,label='Y Poly')
##    plt.legend()
##    plt.subplot(132)
##    plt.plot(x_axis,cent_dat[:,2],label='X box')
##    plt.plot(x_axis,cent_dat[:,3],label='Y box')
##    plt.plot(x_axis,x_pol,label='X Poly')
##    plt.plot(x_axis,y_pol,label='Y Poly')
##    plt.legend()
##    plt.subplot(133)
##    plt.plot(x_axis,cent_dat[:,4],label='W box')
##    plt.plot(x_axis,cent_dat[:,5],label='H box')
##    plt.plot(x_axis,w_pol,label='W Poly')
##    plt.plot(x_axis,h_pol,label='H Poly')
##    plt.legend()
    
    # Prepare data with smoothed centring data
    x_prep = np.zeros((slices,256,256),'uint8')
    y_prep = np.zeros((slices,256,256),'uint8')
    restore_data = []
    for j in range(slices):
        x_prep[j], y_prep[j], rest = mkc.prepare_images_2(x[j],y[j],x_adj_pol[j],y_adj_pol[j],
                                                          x_pol[j],y_pol[j],w_pol[j],h_pol[j])
        restore_data.append(rest)

##    # Save processed data
    with open('D:\DATA_KITS2019\kits19\prep_data2\case_00%.3i.pickle'%i, 'wb') as f:
        pickle.dump([x_prep,y_prep,restore_data], f)

##    np.savez_compressed('D:\DATA_KITS2019\kits19\prep_data3\case_00%.3i.npz'%i,
##                        x=x_prep,y=y_prep,r=rest)
##    np.save('D:\\DATA_KITS2019\\kits19\\prep_data3\\seg_00%.3i.npy'%i,y_prep)
##    np.save('D:\\DATA_KITS2019\\kits19\\prep_data3\\vol_00%.3i.npy'%i,x_prep)
    
    print('Processing time = ', time()-start_time)

##plt.show()
##    img_rest, seg_rest = mkc.restore_images(img_prep,seg_prep,rest)   
##    seg_diff = seg_orig - seg_rest

