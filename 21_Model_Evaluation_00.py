from starter_code import myEvaluationMetrics as mem
from time import time
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
##from sklearn.metrics import accuracy_score, f1_score, precision_score,
from sklearn.metrics import precision_recall_fscore_support
precisions, recalls, f1_scores, supports = [], [], [], []
Overall_f1_scores = []
##poor = 0
##cases = [0]
##for i in cases:
for i in range(210):
    start_time = time()
    print('case_00%.3i'%i)
    # Load processed data
    with open('D:\DATA_KITS2019\kits19\prep_data2\case_00%.3i.pickle'%i, 'rb') as f:
        x, y_true, restore_dat = pickle.load(f)
    rest_dat = np.array(restore_dat)
##    print('Y true shape: ', y_true.shape)
##    file = 'D:\DATA_KITS2019\PredictionsVV\case_00%.3i.npy'%i
##    y_pred = np.load(file).astype('uint8')
    with open('D:\DATA_KITS2019\PredictionsVV\case_00%.3i.pickle'%i, 'rb') as f:
        _, y_pred, _ = pickle.load(f)
##    print('Y pred shape: ', y_pred.shape)
##    f1 = f1_score(y_true.ravel(),y_pred.ravel(),average=None)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true.ravel(),y_pred.ravel(),average=None)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)
    supports.append(sup)
##    m1, m2 = mem.rates(y_true.ravel(),y_pred.ravel(),[1,2])
    Overall_f1_scores.append(np.mean((f1[1],f1[2])))
    print('Case_00%.3i\tF1 Score:\t'%i,f1)
##    if f1[2] < 0.2:
##        poor += 1
##        for j in range(0,x.shape[0],5):
##            path = 'D:\DATA_KITS2019\plots_17\case_00%.3i'%i
##            if not os.path.exists(path):
##                os.makedirs(path)
##            imfilename = 'image_%.6i.png' % j
##            impath = os.path.join(path, imfilename)
##            for k in range(2):
##                plt.subplot(1,2,k+1)
##                plt.imshow(x[j],'gray'),plt.axis('off')
##                if k == 0:
##                    plt.imshow(y_true[j],'jet',alpha=0.5,vmin=0,vmax=2),plt.axis('off')
##                if k == 1:
##                    plt.imshow(y_pred[j],'jet',alpha=0.5,vmin=0,vmax=2),plt.axis('off')
##            plt.savefig(impath,dpi=200)                
            
    print('Processing time = ', time()-start_time)

precisions = np.vstack(precisions)
recalls = np.vstack(recalls)
f1_scores = np.vstack(f1_scores)
supports = np.vstack(supports)

mean_f1 = f1_scores.mean(axis=0)
std_f1 = f1_scores.std(axis=0)
Overall_f1_scores = np.vstack(Overall_f1_scores)
mean_of1 = Overall_f1_scores.mean(axis=0)
std_of1 = Overall_f1_scores.std(axis=0)
print('Overall F1 Score:\t',mean_of1)
