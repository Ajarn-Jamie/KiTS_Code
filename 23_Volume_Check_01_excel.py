##https://www.programcreek.com/python/example/88831/skimage.measure.regionprops
from time import time
import pickle, os, xlrd, xlsxwriter
import numpy as np
from skimage import measure

def check_sphericity(volume):
    verts,faces,_,_ = measure.marching_cubes_lewiner(volume)
    surface_area = measure.mesh_surface_area(verts, faces)
    vol = np.count_nonzero(x)
    sphericity = ((np.pi**(1/3))*((6*vol)**(2/3)))/surface_area
    return sphericity
    
read_workbook = xlrd.open_workbook('KITS19_PreAnalysis2.xlsx')
sheet = read_workbook.sheet_by_index(0)
thicknesses = sheet.col(4)
spacings = sheet.col(5)

write_workbook = xlsxwriter.Workbook('Kidney_Tumor_Analysis_test.xlsx')
worksheet = write_workbook.add_worksheet()
worksheet.write('A1', 'Case')
worksheet.write('B1', 'K or T')
worksheet.write('C1', 'Region Index')
worksheet.write('D1', 'Area')
worksheet.write('E1', 'Volume')
worksheet.write('F1', 'Cent (f)')
worksheet.write('G1', 'Cent (x)')
worksheet.write('H1', 'Cent (y)')
worksheet.write('I1', 'Eq. Dia.')
worksheet.write('J1', 'L.Cent (f)')
worksheet.write('K1', 'L.Cent (x)')
worksheet.write('L1', 'L.Cent (y)')
worksheet.write('M1', 'Maj.Ax.')
worksheet.write('N1', 'Min.Ax.')
worksheet.write('O1', 'Thickness')
worksheet.write('P1', 'Solidity')
worksheet.write('Q1', 'Inert.F.')
worksheet.write('R1', 'Inert.X.')
worksheet.write('S1', 'Inert.Y.')
worksheet.write('T1', 'Moments')
worksheet.write('U1', 'Sphericity')

idx = 0

tum_regions, kid_regions = [], []
##cases = [2]
##for i in cases:
for i in range(210):
    start_time = time()
    print('case_00%.3i'%i)
    with open('D:\DATA_KITS2019\kits19\prep_data2\case_00%.3i.pickle'%i, 'rb') as f:
        x, y_true, restore_dat = pickle.load(f)

    thickness = thicknesses[i+1].value
##    spacing = spacings[i+1].value
    voxel_vol = thickness * 1

    tum_labels = measure.label((y_true == 2).astype('uint8'))
    tum_properties = measure.regionprops(tum_labels)
    tum_regions.append(len(tum_properties))
    for prop in tum_properties:
        worksheet.write('A'+str(idx+2), "%.3i"%i)
        worksheet.write('B'+str(idx+2), "Tumor")
        worksheet.write('C'+str(idx+2), prop.label)
        worksheet.write('D'+str(idx+2), prop.area)
        worksheet.write('E'+str(idx+2), prop.area*voxel_vol)
        worksheet.write('F'+str(idx+2), prop.centroid[0]/y_true.shape[0])
        worksheet.write('G'+str(idx+2), prop.centroid[1]/y_true.shape[1])
        worksheet.write('H'+str(idx+2), prop.centroid[2]/y_true.shape[2])
        worksheet.write('I'+str(idx+2), prop.equivalent_diameter)
        worksheet.write('J'+str(idx+2), prop.local_centroid[0]/y_true.shape[0])
        worksheet.write('K'+str(idx+2), prop.local_centroid[1]/y_true.shape[1])
        worksheet.write('L'+str(idx+2), prop.local_centroid[2]/y_true.shape[2])
        worksheet.write('M'+str(idx+2), prop.major_axis_length)
        worksheet.write('N'+str(idx+2), prop.minor_axis_length)
        worksheet.write('O'+str(idx+2), prop.coords[:,0].max(axis=0) \
                        - prop.coords[:,0].min(axis=0))        
##        worksheet.write('P'+str(idx+2), prop.solidity)
        worksheet.write('Q'+str(idx+2), prop.inertia_tensor_eigvals[0])
        worksheet.write('R'+str(idx+2), prop.inertia_tensor_eigvals[1])
        worksheet.write('S'+str(idx+2), prop.inertia_tensor_eigvals[2])
        worksheet.write('T'+str(idx+2), np.sum(prop.moments))

        x = np.zeros(y_true.shape)
        x[tum_labels==prop.label] = 1
        check = check_sphericity(x)
        worksheet.write('U'+str(idx+2), check)
        print('Tumor Sphericity: ', check)
        
        idx += 1    
    
    kid_labels = measure.label((y_true == 1).astype('uint8'))
    kid_properties = measure.regionprops(kid_labels)
    kid_regions.append(len(kid_properties))
    for prop in kid_properties:
        worksheet.write('A'+str(idx+2), "%.3i"%i)
        worksheet.write('B'+str(idx+2), "Kidney")
        worksheet.write('C'+str(idx+2), prop.label)
        worksheet.write('D'+str(idx+2), prop.area)
        worksheet.write('E'+str(idx+2), prop.area*voxel_vol)
        worksheet.write('F'+str(idx+2), prop.centroid[0]/y_true.shape[0])
        worksheet.write('G'+str(idx+2), prop.centroid[1]/y_true.shape[1])
        worksheet.write('H'+str(idx+2), prop.centroid[2]/y_true.shape[2])
        worksheet.write('I'+str(idx+2), prop.equivalent_diameter)
        worksheet.write('J'+str(idx+2), prop.local_centroid[0]/y_true.shape[0])
        worksheet.write('K'+str(idx+2), prop.local_centroid[1]/y_true.shape[1])
        worksheet.write('L'+str(idx+2), prop.local_centroid[2]/y_true.shape[2])
        worksheet.write('M'+str(idx+2), prop.major_axis_length)
        worksheet.write('N'+str(idx+2), prop.minor_axis_length)
        worksheet.write('O'+str(idx+2), prop.coords[:,0].max(axis=0) \
                        - prop.coords[:,0].min(axis=0))
##        worksheet.write('P'+str(idx+2), prop.solidity)
        worksheet.write('Q'+str(idx+2), prop.inertia_tensor_eigvals[0])
        worksheet.write('R'+str(idx+2), prop.inertia_tensor_eigvals[1])
        worksheet.write('S'+str(idx+2), prop.inertia_tensor_eigvals[2])
        worksheet.write('T'+str(idx+2), np.sum(prop.moments))

        x = np.zeros(y_true.shape)
        x[kid_labels==prop.label] = 1
        check = check_sphericity(x)
        worksheet.write('U'+str(idx+2), check)
##        print('Kidney Sphericity: ', check)

        idx += 1
        
write_workbook.close()
