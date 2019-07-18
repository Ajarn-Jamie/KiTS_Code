from starter_code.utils import load_case
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.measure import regionprops

def centre_image(im, x_adj, y_adj):
    if x_adj > 0:
        im2 = np.pad(im, ((x_adj, 0), (0, 0)), mode='constant')
        im2 = im2[:im.shape[0]-x_adj, :]
    else:
        im2 = np.pad(im, ((0, -x_adj), (0, 0)), mode='constant')
        im2 = im2[-x_adj:, :]

    if y_adj > 0:
        im2 = np.pad(im2, ((0, 0), (y_adj, 0)), mode='constant')
        im2 = im2[:, :im.shape[0]-y_adj]
    else:
        im2 = np.pad(im2, ((0, 0), (0, -y_adj)), mode='constant')
        im2 = im2[:, -y_adj:]
    return im2

##apert_ims = [15,18,19,23,25,31,32,40,43,45,48,
##             50,61,64,65,66,81,85,86,94,97,99,
##             102,107,109,111,117,121,123,124,
##             128,131,133,150,163,166,167,168,
##             169,172,180,185,191,192,193,194,
##             199,202] # Clear Circular Aperture

##apert_ims = [48,50,180] # Darker images

apert_ims = [0,2,9,44,47,51,60,103,105,162,168,170,178,
             189,199] # 2, 170,178 - table artefact

apert_ims = [0]

##i = 0
for i in apert_ims:
##for i in range(50,210):
    print("case_00%.3i"%i)
    volume, segmentation = load_case("case_00%.3i"%i)
    x = volume.get_data().astype('float32')
    y = segmentation.get_data()
    y_mean = np.mean(y, axis=(1,2))
##    max_slice = np.argmax(y_mean)
    max_slice = 120
    img = x[max_slice]
    # Resize to 512x512 if necessary
    if img.shape != (512,512):
        img = cv2.resize(img,(512,512))
    # Convert to 8-bit image
    scaled_data = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    # Equalize histogram
    equal_hist = cv2.equalizeHist(scaled_data)
    # Threshold
    ret1, mask = cv2.threshold(equal_hist,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Filter thresholded mask
    med = cv2.medianBlur(mask,9,None)
    th = cv2.blur(med,(15,15),None)
    # Flood-fill black holes in foreground (from top, bottom-left and bottom-right)
    ff = th.copy()
    cv2.floodFill(ff,None,(0,0),2)
    cv2.floodFill(ff,None,(9,500),2)
    cv2.floodFill(ff,None,(500,500),2)
    ff = np.clip(2 - ff,0,1)
    # Opening to remove foreground objects   
    op = cv2.morphologyEx(ff,cv2.MORPH_OPEN,np.ones((99,99),'uint8'))
    # Apply mask to volume image
    equal_hist *= op
    # Centre image
    center = regionprops(op, equal_hist)[0].centroid
    x_adj = int(scaled_data.shape[0]//2-center[0])
    y_adj = int(scaled_data.shape[1]//2-center[1])
    cimg = centre_image(equal_hist, x_adj, y_adj)
    cimg = cv2.resize(cimg,(equal_hist.shape[1],equal_hist.shape[0]))
    # Bounding box
    cent = cimg.copy()
    contours,hierarchy = cv2.findContours(cimg, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
##    print('Contours: ', len(contours))
    x,y,w,h = cv2.boundingRect(contours[-1])
##    cv2.rectangle(cent,(x,y),(x+w,y+h),(255),3)
    # Zoom to box
    zoom = cimg[y:y+h,x:x+w]
    zoom = cv2.resize(zoom,(512,512))
    # Equalize histogram again
    zoom = cv2.equalizeHist(zoom)

    # Plot images
    plt.subplot(2,4,1),plt.imshow(scaled_data,'gray'),plt.axis('off'),plt.title('Original')
    plt.subplot(2,4,2),plt.imshow(equal_hist,'gray'),plt.axis('off'),
    plt.title('Eq+Thresh\n+Mask+COM')
    plt.scatter(center[1],center[0],c='r',s=75,marker='+')
    plt.subplot(2,4,3),plt.imshow(cimg,'gray'),plt.axis('off'),plt.title('Centered')
    plt.scatter(256,256,c='g',s=75,marker='+')
    plt.subplot(2,4,4),plt.imshow(zoom,'gray'),plt.axis('off'),plt.title('Zoomed')
    plt.subplot(2,4,5),plt.imshow(mask,'gray'),plt.axis('off'),plt.title('Thresh: %.1f'%ret1)
    plt.subplot(2,4,6),plt.imshow(th,'gray'),plt.axis('off'),plt.title('Thresh+Med+Blur')
    plt.subplot(2,4,7),plt.imshow(ff,'gray'),plt.axis('off'),plt.title('Flood-Filled')
    plt.subplot(2,4,8),plt.imshow(op,'gray'),plt.axis('off'),plt.title('Opened: K99')

    
##    plt.savefig('D:\DATA_KITS2019\plots_8\Test01_Case%.3i.png' % i,dpi=600)
##    plt.clf()

plt.show()
