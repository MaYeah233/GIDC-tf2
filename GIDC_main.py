# -*- coding: utf-8 -*-
"""
By Fei Wang, Jan 2022
Contact: WangFei_m@outlook.com
This code implements the ghost imaging reconstruction using deep neural network constraint (GIDC) algorithm
reported in the paper: 
Fei Wang et al. 'Far-field super-resolution ghost imaging with adeep neural network constraint'. Light Sci Appl 11, 1 (2022).  
https://doi.org/10.1038/s41377-021-00680-w
Please cite our paper if you find this code offers any help.

Inputs:
A_real: illumination patterns (pixels * pixels * pattern numbers)
y_real: single pixel measurements (pattern numbers)

Outputs:
x_out: reconstructed image by GIDC (pixels * pixels)
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from GIDC_model_Unet import GIDC_model_Unet
from PIL import Image
import os

# load data
data = loadmat('data.mat') 
result_save_path = '.\\results\\'

# create results save path
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path) 

# define optimization parameters
img_W = 64
img_H = 64
SR = 0.1                                      # sampling rate
lr0 = 0.05                                    # learning rate
TV_strength = 1e-9                            # regularization parameter of Total Variation
num_patterns = int(np.round(img_W*img_H*SR))  # number of measurement times  
Steps = 201                                   # optimization steps

A_real = data['patterns'][:, :, 0:num_patterns]  # illumination patterns
y_real = data['measurements'][0:num_patterns]    # intensity measurements

if (num_patterns > np.shape(data['patterns'])[-1]):
    raise Exception('Please set a smaller SR')

# DGI reconstruction
print('DGI reconstruction...')
B_aver  = 0
SI_aver = 0
R_aver = 0
RI_aver = 0
count = 0
for i in range(num_patterns):    
    pattern = data['patterns'][:,:,i]
    count = count + 1
    B_r = data['measurements'][i]

    SI_aver = (SI_aver * (count -1) + pattern * B_r)/count
    B_aver  = (B_aver * (count -1) + B_r)/count
    R_aver = (R_aver * (count -1) + np.sum(pattern))/count
    RI_aver = (RI_aver * (count -1) + np.sum(pattern)*pattern)/count
    DGI = SI_aver - B_aver / R_aver * RI_aver
# DGI[DGI<0] = 0
print('Finished')

# Build the DNN model
model = GIDC_model_Unet()

# Define the optimizer and learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr0,
    decay_steps=100,
    decay_rate=0.90)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5, beta_2=0.9, epsilon=1e-08)

# Prepare data
y_real = np.reshape(y_real, [1, 1, 1, num_patterns])
A_real = np.reshape(A_real, [1, img_W, img_H, num_patterns])
DGI = np.reshape(DGI, [1, img_W, img_H, 1])

# Preprocessing
DGI = (DGI - np.mean(DGI)) / np.std(DGI)
y_real = (y_real - np.mean(y_real)) / np.std(y_real)
A_real = (A_real - np.mean(A_real)) / np.std(A_real)

# Convert to tensors
y_real = tf.constant(y_real, dtype=tf.float32)
A_real = tf.constant(A_real, dtype=tf.float32)
inpt_temp = tf.constant(DGI, dtype=tf.float32)

# Prepare for surveillance
DGI_temp0 = np.reshape(DGI, [img_W, img_H])
y_real_temp = np.reshape(y_real, [num_patterns])

@tf.function
def train_step(inpt, real_A, real_y):
    with tf.GradientTape() as tape:
        x_pred, y_pred = model([inpt, real_A], training=True)
        
        # define the loss function
        loss_y = tf.reduce_mean(tf.square(real_y - y_pred))
        TV_reg = TV_strength * tf.image.total_variation(x_pred)
        loss = loss_y + TV_reg

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_y, x_pred, y_pred

print('GIDC reconstruction...')

for step in range(Steps):
    train_y_loss, x_out, y_out = train_step(inpt_temp, A_real, y_real)
    
    if step % 100 == 0:
        lr_temp = optimizer._decayed_lr(tf.float32)
        print('step:%d----y loss:%f----learning rate:%f----num of patterns:%d' % (step, train_y_loss, lr_temp, num_patterns))

        x_out = np.reshape(x_out.numpy(), [img_W, img_H])
        y_out = np.reshape(y_out.numpy(), [num_patterns])

        plt.subplot(141)
        plt.imshow(DGI_temp0)
        plt.title('DGI')
        plt.yticks([])

        plt.subplot(142)
        plt.imshow(x_out)
        plt.title('GIDC')
        plt.yticks([])

        ax1 = plt.subplot(143)
        plt.plot(y_out)
        plt.title('pred_y')
        ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
        plt.yticks([])

        ax2 = plt.subplot(144)
        plt.plot(y_real_temp)
        plt.title('real_y')
        ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')
        plt.yticks([])

        plt.subplots_adjust(hspace=0.25, wspace=0.25)
        plt.show()

        x_out = x_out - np.min(x_out)
        x_out = x_out * 255 / np.max(x_out)
        x_out_img = Image.fromarray(x_out.astype('uint8')).convert('L')
        x_out_img.save(result_save_path + 'GIDC_%d_%d.bmp' % (num_patterns, step))

print('Finished!')
