from starter_code.utils import load_case
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.measure import regionprops

def centre_image(im, x_adj, y_adj):
    if x_adj > 0:
        centred_im = np.pad(im, ((x_adj, 0), (0, 0)), mode='constant')
        centred_im = centred_im[:im.shape[0]-x_adj, :]
    else:
        centred_im = np.pad(im, ((0, -x_adj), (0, 0)), mode='constant')
        centred_im = centred_im[-x_adj:, :]

    if y_adj > 0:
        centred_im = np.pad(centred_im, ((0, 0), (y_adj, 0)), mode='constant')
        centred_im = centred_im[:, :im.shape[0]-y_adj]
    else:
        centred_im = np.pad(centred_im, ((0, 0), (0, -y_adj)), mode='constant')
        centred_im = centred_im[:, -y_adj:]
    return centred_im

def restore_image(im, x_adj, y_adj):
    if x_adj > 0:
        centred_im = np.pad(im, ((x_adj, 0), (0, 0)), mode='constant')
        centred_im = centred_im[:im.shape[0], :]
    else:
        centred_im = np.pad(im, ((0, -x_adj), (0, 0)), mode='constant')
        centred_im = centred_im[-x_adj:, :]

    if y_adj > 0:
        centred_im = np.pad(centred_im, ((0, 0), (y_adj, 0)), mode='constant')
        centred_im = centred_im[:, :im.shape[0]]
    else:
        centred_im = np.pad(centred_im, ((0, 0), (0, -y_adj)), mode='constant')
        centred_im = centred_im[:, -y_adj:]
    return centred_im

apert_ims = [1]

##i = 0
##for i in apert_ims:
for i in range(0,10):
    print("case_00%.3i"%i)
    volume, segmentation = load_case("case_00%.3i"%i)
    x = volume.get_data().astype('float32')
    y = segmentation.get_data()
    y_mean = np.mean(y, axis=(1,2))
    max_slice = np.argmax(y_mean)

    img = x[max_slice]
    seg = y[max_slice]
    
    img_norm = img.copy()
    seg_norm = seg.copy()
    
    h_orig,w_orig = img.shape
    # Resize to 512x512 if necessary
    if img.shape != (512,512):
        img = cv2.resize(img,(512,512))
        seg = cv2.resize(seg,(512,512))
##    seg_norm = seg.copy()
    # Convert to 8-bit image
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
##    img_norm = img.copy()
    # Equalize histogram
    img = cv2.equalizeHist(img)
    # Threshold
    ret1, mask = cv2.threshold(img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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
    img *= op
    # Centre image
    center = regionprops(op, img)[0].centroid
    x_adj = int(img.shape[0]//2-center[0])
    y_adj = int(img.shape[1]//2-center[1])
    img = centre_image(img, x_adj, y_adj)
    img = cv2.resize(img,(img.shape[1],img.shape[0]))
    seg = centre_image(seg, x_adj, y_adj)
    seg = cv2.resize(seg,(seg.shape[1],seg.shape[0]))
    # Bounding box
    contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[-1])
    # Zoom to box
    img = img[y:y+h,x:x+w]
    img = cv2.resize(img,(512,512))
    seg = seg[y:y+h,x:x+w]
    seg = cv2.resize(seg,(512,512))
    # Equalize histogram again
    img = cv2.equalizeHist(img)


    # Restore original image sizes
    wr, hr = round((512-w)/2), round((512-h)/2)
    img_rest = cv2.resize(img,(w,h))
    img_rest = np.pad(img_rest, ((hr,hr),(wr,wr)), mode='constant')
    img_rest = cv2.resize(img_rest,(w_orig,h_orig))
    img_rest = restore_image(img_rest, -x_adj, -y_adj)
    print(img_rest.shape)
    # Restore segmentation size
    seg_rest = cv2.resize(seg,(w,h))
    seg_rest = np.pad(seg_rest, ((hr,hr),(wr,wr)), mode='constant')
    seg_rest = cv2.resize(seg_rest,(w_orig,h_orig))
    seg_rest = restore_image(seg_rest, -x_adj, -y_adj)
    # Check difference between original and restored segmentation data
    diff = ((np.clip(seg_norm,0,1)) - (np.clip(seg_rest,0,1))/255).astype('uint8')
    err = (diff.sum()/diff.size)*100

    
    # Plot images
##    plt.subplot(221),plt.imshow(img_norm,'gray'),plt.axis('off'),plt.title('Original')
##    plt.imshow(seg_norm,'jet',alpha=0.2,vmax=2),plt.axis('off')
##    plt.subplot(222),plt.imshow(img,'gray'),plt.axis('off'), plt.title('Prepared')
##    plt.imshow(seg,'jet',alpha=0.2,vmax=2),plt.axis('off')
##    plt.subplot(223),plt.imshow(img_rest,'gray'),plt.axis('off'), plt.title('Restored')
##    plt.imshow(seg_rest,'jet',alpha=0.2,vmax=2),plt.axis('off')
##    plt.subplot(224),plt.imshow(diff,'gray'),plt.axis('off'),plt.title('Seg.Err: %.3f%%'%err)
    
##    plt.savefig('D:\DATA_KITS2019\plots_10\Test01_Case%.3i.png' % i,dpi=600)
##    plt.clf()
    print('X-adj = ',x_adj)
    print('Y-adj = ',y_adj)
    print('W = ',w)
    print('H = ',h)
    print('Error = ',err)
##plt.show()
