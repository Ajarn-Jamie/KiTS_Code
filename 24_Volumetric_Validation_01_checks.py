##https://www.programcreek.com/python/example/88831/skimage.measure.regionprops
import pickle, os, xlrd
import numpy as np
from skimage import measure
from sklearn.metrics import f1_score

read_workbook = xlrd.open_workbook('KITS19_PreAnalysis2.xlsx')
sheet = read_workbook.sheet_by_index(0)
thicknesses = sheet.col(4)

f1_scores = []

cases = [178]
for i in cases:
##for i in range(210):
    print('case_00%.3i'%i)
    with open('D:\DATA_KITS2019\kits19\prep_data2\case_00%.3i.pickle'%i, 'rb') as f:
        x, y_true, restore_dat = pickle.load(f)

    file = 'D:\DATA_KITS2019\Predictions00\case_00%.3i.npy'%i
    y_pred = np.load(file).astype('uint8')
    y_pred_vv = y_pred.copy()
    
    thickness = thicknesses[i+1].value
    voxel_vol = thickness * 1

    kid_labels = measure.label((y_pred == 1).astype('uint8'))
    kid_properties = measure.regionprops(kid_labels)
    
    for j, prop in enumerate(kid_properties):
        if (prop.area*voxel_vol) > 15000:
            print('Label: ', prop.label)
            print('Volume: ', prop.area*voxel_vol)
            print('Centroid[0]: ', prop.centroid[0]/y_true.shape[0])
            print('Eq. Diam.: ', prop.equivalent_diameter)
            print('Maj. Ax.: ', prop.major_axis_length)
            print('Min. Ax.: ', prop.minor_axis_length)
        if (prop.area*voxel_vol) < 15000:
            y_pred_vv[kid_labels==prop.label] = 0
        if  (prop.centroid[0]/y_true.shape[0] < 0.2) or \
           (prop.centroid[0]/y_true.shape[0] > 0.8):
            y_pred_vv[kid_labels==prop.label] = 0
        if  (prop.equivalent_diameter < 15) or \
           (prop.equivalent_diameter > 150):
            y_pred_vv[kid_labels==prop.label] = 0
        if  (prop.major_axis_length < 55):
            y_pred_vv[kid_labels==prop.label] = 0
        if  (prop.minor_axis_length < 30):
            y_pred_vv[kid_labels==prop.label] = 0

    f1_1 = f1_score(y_true.ravel(), y_pred.ravel(), average=None)
    f1_2 = f1_score(y_true.ravel(), y_pred_vv.ravel(), average=None)
    f1_scores.append(f1_2)

f1_scores = np.vstack(f1_scores)
##np.save('f1_scores_vvKid.npy', f1_scores)

##for i in range(f1_scores.shape[0]):
##	if f1_scores[i,1] < 0.6:
##		print('Case %.3i: '%i,f1_scores[i])		
'''
Case 024:  [0.99849819 0.48589901 0.54654803]
Case 178:  [0.99325093 0.46438788 0.64202784]
Case 182:  [0.99870311 0.4398642  0.02301396]
'''
