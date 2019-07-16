import numpy as np
import matplotlib.pyplot as plt
import cv2

class_1_metrics = np.load('Class1_Metrics.npy')
class_2_metrics = np.load('Class2_Metrics.npy')

c1_tpr = class_1_metrics[:,0]
c1_fpr = class_1_metrics[:,1]
c1_tnr = class_1_metrics[:,2]
c1_fnr = class_1_metrics[:,3]

c2_tpr = class_2_metrics[:,0]
c2_fpr = class_2_metrics[:,1]
c2_tnr = class_2_metrics[:,2]
c2_fnr = class_2_metrics[:,3]

overall_tpr = np.mean(np.hstack([c1_tpr.reshape(-1,1),c2_tpr.reshape(-1,1)]),axis=1)
overall_fpr = np.mean(np.hstack([c1_fpr.reshape(-1,1),c2_fpr.reshape(-1,1)]),axis=1)
overall_tnr = np.mean(np.hstack([c1_tnr.reshape(-1,1),c2_tnr.reshape(-1,1)]),axis=1)
overall_fnr = np.mean(np.hstack([c1_fnr.reshape(-1,1),c2_fnr.reshape(-1,1)]),axis=1)

mean_overall_tpr = overall_tpr.mean()
mean_overall_fpr = overall_fpr.mean()
mean_overall_tnr = overall_tnr.mean()
mean_overall_fnr = overall_fnr.mean()

std_overall_tpr = overall_tpr.std()
std_overall_fpr = overall_fpr.std()
std_overall_tnr = overall_tnr.std()
std_overall_fnr = overall_fnr.std()

print('Overall TPR\tMean = %.4f\tStD = %.4f'%(mean_overall_tpr,std_overall_tpr))
print('Overall FPR\tMean = %.4f\tStD = %.4f'%(mean_overall_fpr,std_overall_fpr))
print('Overall TNR\tMean = %.4f\tStD = %.4f'%(mean_overall_tnr,std_overall_tnr))
print('Overall FNR\tMean = %.4f\tStD = %.4f'%(mean_overall_fnr,std_overall_fnr))

