import pickle, os, xlrd
import numpy as np
from skimage import measure
from itertools import groupby, count

def tum_centre_inside_bboxes(centroid, bbox_list):
    for bbox in bbox_list:
        if (centroid[0] >= bbox[0] and centroid[0] <= bbox[3]) \
           and (centroid[1] >= bbox[1] and centroid[1] <= bbox[4]) \
           and (centroid[2] >= bbox[2] and centroid[2] <= bbox[5]):
            inside_check = True
##            print('inside')
            break
        else:
##            print('outside')
            inside_check = False
    return inside_check
    
def bboxes_intersect(tum_bbox, bbox_list):
    check = False
    for bbox in bbox_list:
        dz = min(tum_bbox[3],bbox[3]) - max(tum_bbox[0],bbox[0])
        dx = min(tum_bbox[4],bbox[4]) - max(tum_bbox[1],bbox[1])
        dy = min(tum_bbox[5],bbox[5]) - max(tum_bbox[2],bbox[2])
        if (dz>=0) and (dx>=0) and (dy>=0):
            check = True
    return check

def check_sphericity(volume):
    verts,faces,_,_ = measure.marching_cubes_lewiner(volume)
    surface_area = measure.mesh_surface_area(verts, faces)
    vol = np.count_nonzero(volume)
    sphericity = ((np.pi**(1/3))*((6*vol)**(2/3)))/surface_area
    return sphericity

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def final_check(data):
    checked_data = data.copy()
    data_im_mean = data.mean(axis=(1,2))
    data_nonzero = data_im_mean.nonzero()

    groups = group_consecutives(data_nonzero[0],1)

    group_lengths = []
    for group in groups:
        group_lengths.append(len(group))

    if len(group_lengths) > 1:
        max_index = group_lengths.index(np.array(group_lengths).max())
        indexes = [i for i,x in enumerate(group_lengths) if i!=max_index]
##        print(indexes)
        for index in indexes:
            checked_data[groups[index],:,:] = 0

    return checked_data

read_workbook = xlrd.open_workbook('KITS19_PreAnalysis3.xlsx')
sheet = read_workbook.sheet_by_index(0)
thicknesses = sheet.col(4)

##cases = [213,215,221,228,247,250,255,274,280,284,286,297]
##cases = [223]

##cases = [213]
##for i in cases:
for i in range(210,300):
    print('case_00%.3i'%i)
    with open('D:\DATA_KITS2019\kits19\prep_data_test\case_00%.3i.pickle'%i, 'rb') as f:
        x, y_true, restore_dat = pickle.load(f)

    file = 'D:\DATA_KITS2019\Predictions01\case_00%.3i.npy'%i
    y_pred = np.load(file).astype('uint8')
    y_pred_vv = y_pred.copy()
    
    thickness = thicknesses[i+1].value
    voxel_vol = thickness * 1
    
    kid_labels = measure.label((y_pred == 1).astype('uint8'))
    kid_properties = measure.regionprops(kid_labels)
    kid_bboxes = []
    kid_centres = []
    for prop in kid_properties:
        if (prop.area*voxel_vol) < 19000:
            y_pred_vv[kid_labels==prop.label] = 0
            continue
        if  (prop.centroid[0]/y_true.shape[0] < 0.2) or \
           (prop.centroid[0]/y_true.shape[0] > 0.8):
            y_pred_vv[kid_labels==prop.label] = 0
            continue
        if  (prop.equivalent_diameter < 20) or \
           (prop.equivalent_diameter > 150):
            y_pred_vv[kid_labels==prop.label] = 0
            continue
        if  (prop.major_axis_length < 55):
            y_pred_vv[kid_labels==prop.label] = 0
            continue
        if  (prop.minor_axis_length < 30):
            y_pred_vv[kid_labels==prop.label] = 0
            continue
        if (prop.coords[:,0].max(axis=0) - prop.coords[:,0].min(axis=0)) < 2:
            y_pred_vv[kid_labels==prop.label] = 0
##            print('Kidney region TOO THIN')

        kid_bboxes.append(prop.bbox)
        kid_centres.append([prop.centroid[0]/y_true.shape[0],
                            prop.centroid[1]/y_true.shape[1],
                            prop.centroid[2]/y_true.shape[2]])   
    
    tum_labels = measure.label((y_pred_vv == 2).astype('uint8'))
    tum_properties = measure.regionprops(tum_labels)
    
    for prop in tum_properties:
        if tum_centre_inside_bboxes(prop.bbox, kid_bboxes):
            replace_val = 1
##            print('Tumor region inside kidney bbox')
        else:
            replace_val = 0
##            y_pred_vv[tum_labels==prop.label] = replace_val
##            print('Tumor region outside kidney bbox')
            
        if (prop.area*voxel_vol) < 350:
            y_pred_vv[tum_labels==prop.label] = replace_val
##            print('volume: ', prop.area*voxel_vol)
            continue
        if  (prop.equivalent_diameter < 5) or \
           (prop.equivalent_diameter > 150):
            y_pred_vv[tum_labels==prop.label] = replace_val
##            print('equivalent diameter')
            continue
        if  (prop.major_axis_length < 10):
            y_pred_vv[tum_labels==prop.label] = replace_val
##            print('major axis')
            continue
        if  (prop.minor_axis_length < 3):
            y_pred_vv[tum_labels==prop.label] = replace_val
##            print('minor axis')
            continue
        if prop.coords[:,0].max(axis=0) - prop.coords[:,0].min(axis=0) < 2:
            y_pred_vv[tum_labels==prop.label] = replace_val
##            print('Tumor region TOO THIN')
            continue

        if  (prop.centroid[0]/y_true.shape[0] < 0.1) or \
           (prop.centroid[0]/y_true.shape[0] > 0.9):
            y_pred_vv[tum_labels==prop.label] = replace_val
##            print('centre frame')
            continue
        tum_centre = np.array([prop.centroid[0]/y_true.shape[0],
                               prop.centroid[1]/y_true.shape[1],
                               prop.centroid[2]/y_true.shape[2]])
        dist = []
        for kid_centre in kid_centres:
            dist.append(np.linalg.norm(np.array(kid_centre)-tum_centre))            
        if np.array(dist).min() > 0.5:
            y_pred_vv[tum_labels==prop.label] = replace_val
##            print('centroid distance too far from kideny(s)')
            continue
        if not bboxes_intersect(prop.bbox, kid_bboxes):
            y_pred_vv[tum_labels==prop.label] = 0
            continue
##            print('No bbox intersection')
        a = np.zeros(y_pred_vv.shape)
        a[tum_labels==prop.label] = 1
        sphericity = check_sphericity(a)
        if (sphericity < 0.29):
            y_pred_vv[tum_labels==prop.label] = replace_val
##            print('Tumor sphericity mismatch: ', sphericity)
            continue
##        else:
##            print('Tumor sphericity OK ', sphericity)            

    y_pred_vv = final_check(y_pred_vv)

    path = 'D:\DATA_KITS2019\Predictions_final2\\'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+'\case_00%.3i.pickle'%i, 'wb') as f:
        pickle.dump([x,y_pred_vv,restore_dat], f)
