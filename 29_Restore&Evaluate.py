from starter_code.utils import load_segmentation
import starter_code.myKitsCode as mkc
from time import time
import cv2
import numpy as np
import pickle
from sklearn.metrics import f1_score

f1_scores = []
##cases = [159]
##for i in cases:
for i in range(210):
    start_time = time()
    print('case_00%.3i'%i)
    # Load original data
    y_true = load_segmentation("case_00%.3i"%i)

    y_true = y_true.get_data().astype('uint8')      
    
    with open('D:\DATA_KITS2019\PredictionsVV\case_00%.3i.pickle'%i, 'rb') as f:
        x_prep, y_pred, restore_dat = pickle.load(f)

    y_restored = np.zeros(y_true.shape,'uint8')
    seg_diff = 0
    for j in range(y_true.shape[0]):      
##        _, y_restored[j] = mkc.restore_images(x_prep[j],y_pred[j],restore_dat[j])
        _, a = mkc.restore_images(x_prep[j],y_pred[j],restore_dat[j])
        y_restored[j] = cv2.resize(a,(y_restored[j].shape[1],y_restored[j].shape[0]))
        seg_diff += np.sum(y_true[j] - y_restored[j])

    f1 = f1_score(y_true.ravel(),y_restored.ravel(),average = None)
    f1_scores.append(f1)
    print('F1 Scores: ',f1)
    print('Processing time = ', time()-start_time)

f1_scores = np.vstack(f1_scores)
print('Mean F1 Scores = ',f1_scores.mean(axis=0))
np.save('F1_scores_restored.npy',f1_scores)
