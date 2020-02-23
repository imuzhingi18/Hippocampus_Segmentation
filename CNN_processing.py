# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:04:00 2020

@author: itaim
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:02:23 2020

@author: Itai Muzhingi
"""
import os
import nibabel as nib
import numpy as np
from skimage.transform import resize
from unet import *
import matplotlib.pyplot as plt

def filenames():
    images = os.listdir('imagesTr')
    labels = os.listdir('labelsTr')
    images.sort()
    labels.sort()
    return images, labels

image_list, label_list = filenames()
parent_dir = os.getcwd()

def get_images(image_list, label_list):
    shape = (32,32,32)
    image_array = []
    label_array = []
    
    for i in range(len(image_list)):
        if image_list[i] == label_list[i]:
            os.chdir('imagesTr')
            image = nib.load(image_list[i])
            resampled_image = resize(image.get_fdata(),shape)
            image_slice_0 = resampled_image[13, :, :]
            image_slice_1 = resampled_image[14, :, :]
            image_slice_2 = resampled_image[15, :, :]
            image_slice_3 = resampled_image[16, :, :]
            image_slice_4 = resampled_image[17, :, :]
            image_slice_5 = resampled_image[18, :, :]
            image_slice_6 = resampled_image[19, :, :]
            #plt.imshow(image_slice, cmap = 'gray')
            #plt.show()
            image_array.append(image_slice_0[:, :, np.newaxis])
            image_array.append(image_slice_1[:, :, np.newaxis])
            image_array.append(image_slice_2[:, :, np.newaxis])
            image_array.append(image_slice_3[:, :, np.newaxis])
            image_array.append(image_slice_4[:, :, np.newaxis])
            image_array.append(image_slice_5[:, :, np.newaxis])
            image_array.append(image_slice_6[:, :, np.newaxis])
            os.chdir(parent_dir)
            os.chdir('labelsTr')
            label = nib.load(label_list[i])
            resampled_label = np.rint(np.clip(resize(label.get_fdata(), shape), 0, 1))
            label_slice_0 = resampled_label[13, :, :]
            label_slice_1 = resampled_label[14, :, :]
            label_slice_2 = resampled_label[15, :, :]
            label_slice_3 = resampled_label[16, :, :]
            label_slice_4 = resampled_label[17, :, :]
            label_slice_5 = resampled_label[18, :, :]
            label_slice_6 = resampled_label[19, :, :]
            #plt.imshow(label_slice, cmap = 'gray')
            #plt.show()
            label_array.append(label_slice_0[:,:,np.newaxis])
            label_array.append(label_slice_1[:,:,np.newaxis])
            label_array.append(label_slice_2[:,:,np.newaxis])
            label_array.append(label_slice_3[:,:,np.newaxis])
            label_array.append(label_slice_4[:,:,np.newaxis])
            label_array.append(label_slice_5[:,:,np.newaxis])
            label_array.append(label_slice_6[:,:,np.newaxis])
            os.chdir(parent_dir)
    return np.array(image_array), np.array(label_array)
            
training_images, training_labels = get_images(image_list, label_list)         
normalized_images = training_images/255.0   
img_train = normalized_images[208:1700,:,:,:] #208:
seg_train = training_labels[208:1700,:,:,:]
img_val = normalized_images[:208,:,:,:]
seg_val = training_labels[:208,:,:,:]
img_val_2 = normalized_images[1700:,:,:,:]
seg_val_2 = training_labels[1700:,:,:,:]
print(img_val_2.shape)
print(seg_val_2.shape)
print(img_train.shape)
model = get_unet(input_dim = (32, 32, 1),output_dim = (32, 32, 1), num_output_classes=1)


history = model.fit(x = img_train,
                    y = seg_train,
                    batch_size = 32,  #50 then 4, 10
                    epochs = 1,
                    verbose = 1,
                    validation_data = (img_val, seg_val)) 

os.chdir(parent_dir)
model.save_weights('model_weights_best_1.h5')


os.chdir('imagesTs')
test_array = []
test_list = os.listdir()
for i in range(len(test_list)):
    testing_image = nib.load(test_list[i])
    testing_image = resize(testing_image.get_fdata(),(32,32,32))
    image_slice = testing_image[16, :, :]
    test_array.append(image_slice[:, :, np.newaxis])
final_test = np.array(test_array)
final_test = final_test/255.0
prediction = model.predict(final_test)
print(prediction.shape)

for i in range(len(prediction)):
    plt.imshow(np.squeeze(final_test[i]), cmap='gray')
    plt.show()
    image = np.squeeze(prediction[i], axis = 2)
    plt.imshow(image, cmap = 'jet')
    plt.show()           
    
    
os.chdir(parent_dir)
new_model = get_unet(input_dim = (32, 32, 1),output_dim = (32, 32, 1), num_output_classes=1)
new_model.summary()
new_model.load_weights('model_weights_best_1.h5')

pred_val = new_model.predict(img_val_2)
for i in range(len(pred_val)):
    plt.imshow(np.squeeze(img_val_2[i]), cmap = 'gray')
    plt.show()
    plt.imshow(np.squeeze(seg_val_2[i]), cmap='gray')
    plt.show()
    new_pred = np.squeeze(pred_val[i], axis = 2)
    plt.imshow(np.rint(np.clip(new_pred, 0,1)))
    plt.show()
    plt.imshow(new_pred, cmap = 'jet')
    plt.show() 
    
    
plt.plot(history.history['loss'], label = 'Training Loss', linestyle = '--')
plt.title('Training Loss', fontweight = 'bold', fontsize = 18)
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.xlabel('Epoch Number', fontweight = 'bold', fontsize = 15)
plt.ylabel('Loss', fontweight = 'bold', fontsize = 15)
plt.legend(loc='upper right')
plt.grid(True, color='#999999', linestyle='--', alpha=0.2)
plt.show()


plt.plot(history.history['acc'], color = 'black', label = 'Training Accuracy')
plt.title('Training Accuracy', fontweight = 'bold', fontsize = 18)
plt.plot(history.history['val_acc'], color = 'red', label = 'Validation Accuracy', linestyle = '--')
plt.xlabel('Epoch Number', fontweight = 'bold', fontsize = 15)
plt.ylabel('Accuracy', fontweight = 'bold', fontsize = 15)
plt.legend(loc='lower right')
plt.grid(True, color='#999999', linestyle='--', alpha=0.2)
plt.show()

def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


list_dice = []
for i in range(len(pred_val)):
    val_label = np.squeeze(seg_val_2[i])
    val_label = np.array(val_label, dtype='f')
    pred_label = np.squeeze(pred_val[i], axis = 2)
    pred_label = np.array(pred_label, dtype='f')
    list_dice.append(dice(val_label,pred_label))
plt.hist(list_dice, color = 'brown', edgecolor='black', linewidth=1.2)
plt.title('Dice Score Distribution on Test Data', fontweight = 'bold', fontsize = 18)
plt.xlabel('Dice Score', fontweight = 'bold', fontsize = 15)
plt.ylabel('Frequency', fontweight = 'bold', fontsize = 15)
plt.grid(True, color='#999999', linestyle='--', alpha=0.2)
