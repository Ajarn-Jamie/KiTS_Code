import numpy as np
import cv2
from skimage.measure import regionprops

def fill_segments(seg_data):
    filled_data = np.zeros(seg_data.shape,'uint8')
    for i in range(seg_data.shape[0]):
        im_floodfill = seg_data[i].copy()
        cv2.floodFill(im_floodfill,None,(0,0),2)
        im_floodfill = np.clip(2 - im_floodfill,0,2)
        indexes = im_floodfill == 2
        filled = seg_data[i].copy()
        filled[indexes] += 1
        filled_data[i] = filled
    return filled_data    


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
##    print('done')
    return centred_im


def prepare_images_1(img,seg):
    h_orig, w_orig = img.shape
    # Resize to 512x512 if necessary
    if (h_orig,w_orig) != (512,512):
        img = cv2.resize(img,(512,512))
        seg = cv2.resize(seg,(512,512))
    # Convert to 8-bit image
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    # Equalize histogram
    img = cv2.equalizeHist(img)
    # Threshold
    ret1, mask = cv2.threshold(img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Filter thresholded mask
    med = cv2.medianBlur(mask,9,None)
    th = cv2.blur(med,(15,15),None)
    # Flood-fill black holes in foreground (from top-left, bottom-left and bottom-right)
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
    # Save restore data
    rest = [-x_adj, -y_adj, w, h, w_orig, h_orig]
    return img,seg,rest


def restore_images(img,seg,rest):
    x_adj, y_adj, w, h, w_orig, h_orig = rest
    w = np.clip(w,0,w_orig).astype('int')
    h = np.clip(h,0,h_orig).astype('int')
    # Restore original image sizes
    wr, hr = int(round((w_orig-w)/2)), int(round((h_orig-h)/2))
    img_rest = cv2.resize(img,(w,h))
    img_rest = np.pad(img_rest, ((hr,hr),(wr,wr)), mode='constant')
    img_rest = cv2.resize(img_rest,(w_orig,h_orig))
    img_rest = restore_image(img_rest, x_adj, y_adj)
    # Restore segmentation size
    seg_rest = cv2.resize(seg,(w,h))
    seg_rest = np.pad(seg_rest, ((hr,hr),(wr,wr)), mode='constant')
    seg_rest = cv2.resize(seg_rest,(w_orig,h_orig))
    seg_rest = restore_image(seg_rest, x_adj, y_adj)
    return img_rest, seg_rest


def get_centring_data(img):
    h_orig, w_orig = img.shape
    # Resize to 512x512 if necessary
    if (h_orig,w_orig) != (512,512):
        img = cv2.resize(img,(512,512))
    # Convert to 8-bit image
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    # Equalize histogram
    img = cv2.equalizeHist(img)
    # Threshold
    ret1, mask = cv2.threshold(img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Filter thresholded mask
    med = cv2.medianBlur(mask,9,None)
    th = cv2.blur(med,(15,15),None)
    # Flood-fill black holes in foreground (from top-left, bottom-left and bottom-right)
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
    # Bounding box
    contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[-1])   
    return x_adj, y_adj, x, y, w, h


def prepare_images_2(img,seg,x_adj,y_adj, x, y, w, h):
    h_orig, w_orig = img.shape
    # Resize to 512x512 if necessary
    if (h_orig,w_orig) != (512,512):
        img = cv2.resize(img,(512,512))
        seg = cv2.resize(seg,(512,512))
    # Convert to 8-bit image
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    # Equalize histogram
    img = cv2.equalizeHist(img)
    # Threshold
    ret1, mask = cv2.threshold(img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Filter thresholded mask
    med = cv2.medianBlur(mask,9,None)
    th = cv2.blur(med,(15,15),None)
    # Flood-fill black holes in foreground (from top-left, bottom-left and bottom-right)
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
    x_adj, y_adj = int(x_adj), int(y_adj)
    img = centre_image(img, x_adj, y_adj)
    img = cv2.resize(img,(img.shape[1],img.shape[0]))
    seg = centre_image(seg, x_adj, y_adj)
    seg = cv2.resize(seg,(seg.shape[1],seg.shape[0]))
##    # Bounding box
##    contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
##                                          cv2.CHAIN_APPROX_SIMPLE)
##    x,y,w,h = cv2.boundingRect(contours[-1])
    # Zoom to box
    x,y,w,h = int(x),int(y),int(w),int(h)
    img = img[y:y+h,x:x+w]
##    print(img.shape)
    img = cv2.resize(img,(256,256))
    seg = seg[y:y+h,x:x+w]
    seg = cv2.resize(seg,(256,256))
    # Equalize histogram again
    img = cv2.equalizeHist(img)
    # Save restore data
    rest = [-x_adj, -y_adj, w, h, w_orig, h_orig]
    return img,seg,rest
