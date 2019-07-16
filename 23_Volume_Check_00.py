from time import time
import pickle, os, xlrd
import numpy as np
from skimage import measure

workbook = xlrd.open_workbook('KITS19_PreAnalysis2.xlsx')
sheet = workbook.sheet_by_index(0)
thicknesses = sheet.col(4)
spacings = sheet.col(5)

tum_vols, kid_vols = [], []
##cases = [0]
##for i in cases:
for i in range(1):
    start_time = time()
    print('case_00%.3i'%i)
    # Load processed data
    with open('D:\DATA_KITS2019\kits19\prep_data2\case_00%.3i.pickle'%i, 'rb') as f:
        x, y_true, restore_dat = pickle.load(f)
    rest_dat = np.array(restore_dat)

    thickness = thicknesses[i+1].value
    spacing = spacings[i+1].value

    tumors = y_true == 2
    tumors = tumors.ravel().nonzero()
    tum_vol = tumors[0].shape[0] * thickness * spacing
    print('Tumor volume = ',tum_vol)
    tum_vols.append(tum_vol)

    kids = y_true == 1
    kids = kids.ravel().nonzero()
    kid_vol = kids[0].shape[0] * thickness * spacing
    print('Kidney volume = ',kid_vol)
    kid_vols.append(kid_vol)

    print('Processing time = ', time()-start_time)

tum_vols = np.vstack(tum_vols)
kid_vols = np.vstack(kid_vols)
