#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:14:20 2020

@author: Itai Muzhingi
"""
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.merge import concatenate
from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D

weight_decay = 1e-4


# def get_unet(input_dim, num_output_classes):
def get_unet(input_dim, output_dim, num_output_classes):
    img_input = Input(shape=input_dim, name='image')
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=True)(img_input)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv3', trainable=True)(x1)
    ds1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x1)

    # Block 2
    # input is of size : 128 x 128 x 64
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(ds1)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=True)(x2)
    ds2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x2)

    # Block 3
    # input is of size : 64 x 64 x 128
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=True)(ds2)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=True)(x3)
    ds3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x3)

    # Block 4
    # input is of size : 32 x 32 x 256
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=True)(ds3)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=True)(x4)
    ds4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x4)

    # Bottleneck x5
    x5 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=True)(ds4)
    x5 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=True)(x5)

    # Upsampling 1.
    us1 = concatenate([UpSampling2D(size=(2, 2))(x5), x4])
    x6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='us1_conv1', trainable=True)(us1)
    x6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='us1_conv2', trainable=True)(x6)

    # Upsampling 2.
    us2 = concatenate([UpSampling2D(size=(2, 2))(x6), x3])
    x7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='us2_conv1', trainable=True)(us2)
    x7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='us2_conv2', trainable=True)(x7)

    # Upsampling 3.
    us3 = concatenate([UpSampling2D(size=(2, 2))(x7), x2])
    x8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='us3_conv1', trainable=True)(us3)
    x8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='us3_conv2', trainable=True)(x8)

    # Upsampling 4.
    us4 = concatenate([UpSampling2D(size=(2, 2))(x8), x1])
    x9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='us4_conv1', trainable=True)(us4)
    x9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='us4_conv2', trainable=True)(x9)

    dense_prediction = Conv2D(
        num_output_classes,
        (3, 3),
        padding='same',
        activation='sigmoid',
        kernel_initializer='orthogonal',
        kernel_regularizer=l2(weight_decay),
        bias_regularizer=l2(weight_decay))(x9)

    model = Model(inputs=img_input, outputs=dense_prediction)
    opt = Adam(1e-4)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
