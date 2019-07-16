from keras_deeplab_v3_plus.model import Deeplabv3
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
##import os

deeplab_model = Deeplabv3()
##deeplab_model = Deeplabv3(input_shape=(512,512,3), classes=4, last_activation=True, OS=16) 
##deeplab_model.load_weights('deeplabv3_weights_tf_dim_ordering_tf_kernels.h5', by_name = True)
##deeplab_model = Deeplabv3(input_shape=(512, 512, 1), classes=3, alpha=1)

# Load image data
with open('D:\DATA_KITS2019\kits19\prep_data2\case_00001.pickle', 'rb') as f:
    x_prep, y_prep, restore_dat = pickle.load(f)
##ch = x_prep[255]
##img = np.zeros((512,512,3),'uint8')
##img[:,:,0] = ch
##img[:,:,1] = ch
##img[:,:,2] = ch
##resized = img.copy()

img = plt.imread("keras_deeplab_v3_plus/imgs/image1.jpg")

w, h, _ = img.shape
ratio = 512. / np.max([w,h])
resized = cv2.resize(img,(int(ratio*h),int(ratio*w)))
resized = resized / 127.5 - 1.
pad_x = int(512 - resized.shape[0])
resized2 = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')
res = deeplab_model.predict(np.expand_dims(resized2,0))
labels = np.argmax(res.squeeze(),-1)
##plt.imshow(resized)
plt.imshow(labels[:-pad_x],alpha=.5)
plt.show()
##cases = [203]
##for i in cases:
##for i in range(10,210):
##    start_time = time()
##    print('case_00%.3i'%i)
##    # Load processed data
##    with open('D:\DATA_KITS2019\kits19\prep_data2\case_00%.3i.pickle'%i, 'rb') as f:
##        x_prep, y_prep, restore_dat = pickle.load(f)
##        
##    # Create directory to save frame images
##    im_dir = r'D:\DATA_KITS2019\plots_13\case_00%.3i' % i
##    if not os.path.exists(im_dir):
##        os.makedirs(im_dir)
##
##
##    print('Processing time = ', time()-start_time)


