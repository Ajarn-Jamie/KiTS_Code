from unet.model import unet
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
##import os

# Create instance of U-net
model = unet()

# Load image data
with open('D:\DATA_KITS2019\kits19\prep_data2\case_00001.pickle', 'rb') as f:
    x_prep, y_prep, restore_dat = pickle.load(f)

x_prep = x_prep.reshape(x_prep.shape[0], 512, 512, 1).astype('float32')
x_prep /= 255

y_prep = y_prep.reshape(y_prep.shape[0], 512, 512, 1)

model.fit(x_prep,y_prep,batch_size=1,epochs=10)


####
####img = plt.imread("keras_deeplab_v3_plus/imgs/image1.jpg")
####
####w, h, _ = img.shape
####ratio = 512. / np.max([w,h])
####resized = cv2.resize(img,(int(ratio*h),int(ratio*w)))
####resized = resized / 127.5 - 1.
####pad_x = int(512 - resized.shape[0])
####resized2 = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')
####res = deeplab_model.predict(np.expand_dims(resized2,0))
####labels = np.argmax(res.squeeze(),-1)
######plt.imshow(resized)
####plt.imshow(labels[:-pad_x],alpha=.5)
####plt.show()
######cases = [203]
######for i in cases:
######for i in range(10,210):
######    start_time = time()
######    print('case_00%.3i'%i)
######    # Load processed data
######    with open('D:\DATA_KITS2019\kits19\prep_data2\case_00%.3i.pickle'%i, 'rb') as f:
######        x_prep, y_prep, restore_dat = pickle.load(f)
######        
######    # Create directory to save frame images
######    im_dir = r'D:\DATA_KITS2019\plots_13\case_00%.3i' % i
######    if not os.path.exists(im_dir):
######        os.makedirs(im_dir)
######
######
######    print('Processing time = ', time()-start_time)


