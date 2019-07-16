from time import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom

cases = [15]
for i in cases:
##for i in range(0):
    start_time = time()
    print('case_00%.3i'%i)
    # Load processed data
    with open('D:\DATA_KITS2019\kits19\prep_data2\case_00%.3i.pickle'%i, 'rb') as f:
        x_prep, y, restore_dat = pickle.load(f)
    rest_dat = np.array(restore_dat)

    y = zoom(y,(0.5,0.1,0.1))
    kidneys = y == 1
    tumour = y == 2
    voxels = kidneys | tumour

    a,b,c = voxels.shape
    colors = np.empty((a,b,c,4), dtype=np.float32)
##    colors[kidneys] = 'red'
##    colors[tumour] = 'blue'
    colors[kidneys] = [1,0,0,0.5]
    colors[tumour] = [0,0,1,1.0]
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor=None)
    ax.tick_params(axis='both',labelleft=[],labelright=[],
                   labelbottom=[])
    ax.view_init(elev=-35,azim=-92)
##    ax.view_init(elev=90,azim=0)
    print('Processing time = ', time()-start_time)
##    plt.savefig('case%i_3d.png'%i,dpi=1000)
##    print(ax.azim)
    plt.show()
