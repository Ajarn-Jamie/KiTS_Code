##https://www.programcreek.com/python/example/88831/skimage.measure.regionprops
import pickle, os, xlrd
import numpy as np
from skimage import measure
from sklearn.metrics import f1_score

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

read_workbook = xlrd.open_workbook('KITS19_PreAnalysis2.xlsx')
sheet = read_workbook.sheet_by_index(0)
thicknesses = sheet.col(4)

f1_scores = []
##cases = [15,19,20,37,83,103,120,121,137,164,165,182,188,191,194,198,208]
##cases = [20,120,121,208]
##cases = [7,15,17,20,30,37,60,120,165,178,180,182,188,194,208]
##cases = [120]
##for i in cases:
for i in range(210):
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
            

##    f1_1 = f1_score(y_true.ravel(), y_pred.ravel(), average = None)
##    print('Original F1 = ', f1_1)
    f1_2 = f1_score(y_true.ravel(), y_pred_vv.ravel(), average = None)
    print('Validated F1 = ', f1_2)
##    print('X-shape: ',x.shape)
##
    f1_scores.append(f1_2)

##    path = 'D:\DATA_KITS2019\PredictionsVV\\'
##    if not os.path.exists(path):
##        os.makedirs(path)
##    with open(path+'\case_00%.3i.pickle'%i, 'wb') as f:
##        pickle.dump([x,y_pred_vv,restore_dat], f)

f1_scores = np.vstack(f1_scores)
np.save('f1_scores_vvKT_final.npy',f1_scores)

##for i in range(f1_scores.shape[0]):
##    if f1_scores[i,2] < 0.1:
##        print('Case_%.5i F1: '%i,f1_scores[i])
'''
f1_scores.mean(axis=0)  =>  array([0.99832007, 0.84970132, 0.50193409])
f1_scores.std(axis=0)   =>  array([0.00366223, 0.08567089, 0.26868652])

Case_00007 F1:  [0.99743909 0.87712289 0.        ]
Case_00015 F1:  [0.99280713 0.8659625  0.        ]
Case_00017 F1:  [0.99903704 0.91946135 0.        ]
Case_00020 F1:  [0.99890759 0.84462984 0.        ]
Case_00030 F1:  [0.94843565 0.74364268 0.        ]
Case_00037 F1:  [0.99854716 0.53167827 0.01615555]
Case_00060 F1:  [0.99824669 0.77077325 0.        ]
Case_00120 F1:  [0.9985507  0.82898775 0.        ]
Case_00165 F1:  [0.99984904 0.90457123 0.03753297]
Case_00178 F1:  [0.99132514 0.46902275 0.        ]
Case_00180 F1:  [0.9987964  0.88903542 0.        ]
Case_00182 F1:  [0.99934829 0.62822676 0.0232187 ]
Case_00188 F1:  [0.99925499 0.85233253 0.08421026]
Case_00194 F1:  [0.99926984 0.559872   0.02607578]
Case_00208 F1:  [0.9993573  0.90317298 0.        ]
'''
