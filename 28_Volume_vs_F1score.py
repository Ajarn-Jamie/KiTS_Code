##https://www.programcreek.com/python/example/88831/skimage.measure.regionprops
import pickle, os, xlrd
import numpy as np
from skimage import measure
from sklearn.metrics import f1_score

read_workbook = xlrd.open_workbook('KITS19_PreAnalysis2.xlsx')
sheet = read_workbook.sheet_by_index(0)
thicknesses = sheet.col(4)

##f1_scores = np.load('f1_scores_vvKT_final.npy')

kid_volumes = []
tum_volumes = []

for i in range(210):
    print('case_00%.3i'%i)
    with open('D:\DATA_KITS2019\kits19\prep_data2\case_00%.3i.pickle'%i, 'rb') as f:
        _, y_true, _ = pickle.load(f)
    
    thickness = thicknesses[i+1].value
    voxel_vol = thickness * 1
    
    kid_vol = 0
    tum_vol = 0

    kid_labels = measure.label((y_true == 1).astype('uint8'))
    kid_properties = measure.regionprops(kid_labels)

    for prop in kid_properties:
        kid_vol += (prop.area*voxel_vol)
          
    
    tum_labels = measure.label((y_true == 2).astype('uint8'))
    tum_properties = measure.regionprops(tum_labels)
    
    for prop in tum_properties:
        tum_vol += (prop.area*voxel_vol)

    kid_volumes.append(kid_vol)
    tum_volumes.append(tum_vol)

kid_volumes = np.vstack(kid_volumes)
tum_volumes = np.vstack(tum_volumes)

np.save('kid_volumes.npy',kid_volumes)
np.save('tum_volumes.npy',tum_volumes)
