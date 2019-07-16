from starter_code.utils import load_case
import matplotlib.pyplot as plt
import numpy as np
import cv2

##apert_ims = [15,18,19,23,25,31,32,40,43,45,48,
##             50,61,64,65,66,81,85,86,94,97,99,
##             102,107,109,111,117,121,123,124,
##             128,131,133,150,163,166,167,168,
##             169,172,180,185,191,192,193,194,
##             199,202]
##apert_ims = [48,50,180] # Darker images

apert_ims = [50]

##i = 0
##for i in apert_ims:
for i in range(210):
    print("case_00%.3i"%i)
    volume, segmentation = load_case("case_00%.3i"%i)
    x = volume.get_data().astype('float32')
    y = segmentation.get_data()
    y_mean = np.mean(y, axis=(1,2))
    max_slice = np.argmax(y_mean)

    img = x[max_slice]
    
    scaled_data = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')

    equal_hist = cv2.equalizeHist(scaled_data)
   
    ret1, th1 = cv2.threshold(equal_hist,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)

    plt.subplot(1,3,1),plt.imshow(scaled_data,'gray'),plt.axis('off'),plt.title('8-bit')

    plt.subplot(1,3,2),plt.imshow(equal_hist,'gray'),plt.axis('off'),plt.title('Equalized')

##    plt.subplot(2,3,3),plt.hist(scaled_data.ravel(),256, [0,256], color = 'k')
##    plt.title('Scaled Hist'), plt.ylim([0,20e3])

    plt.subplot(1,3,3),plt.imshow(th1,'gray'),plt.axis('off'),plt.title('Thresh: %f'%ret1)

##    plt.subplot(2,3,5),plt.hist(th1.ravel(),256, [0,256], color = 'k')
##    plt.title('Thresh Hist'), #plt.ylim([0,20e3])


    plt.savefig('D:\DATA_KITS2019\plots_3_aperture\Test02_Case%.3i.png' % i,
                dpi=600)
##plt.show()

##    cimg = 1 - scaled_data
##    cimg = (cimg*255).astype('uint8')
##
##    circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,512,
##                                param1=50,param2=30,minRadius=210,maxRadius=256)
##    ##circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,100)
##    if circles is not None:
##        cimg = np.zeros((512,512,3),'uint8')
##        cimg = cv2.circle(cimg,(circles[0,0,0],circles[0,0,1]),
##                          circles[0,0,2],(0,255,0),3)
##        print('Centre: %.1f,%.1f'%(circles[0,0,0],circles[0,0,1]))
##        print('Radius: %.1f'%circles[0,0,2])
##        plt.subplot(1,3,3),plt.imshow(cimg,'gray'),plt.axis('off')
