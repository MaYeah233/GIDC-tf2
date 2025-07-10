# -*- coding: utf-8 -*-
"""
By Fei Wang, Jan 1, 2022
Contact: WangFei_m@outlook.com
This code implements the DNN structure and measurements process of GIDC algorithm reported in the paper: 
Fei Wang et al. 'Far-field super-resolution ghost imaging with adeep neural network constraint'. Light Sci Appl 11, 1 (2022).  
https://doi.org/10.1038/s41377-021-00680-w
Please cite our paper if you find this code offers any help.

Inputs:
inpt: DGI results (batch_size * pixels * pixels * 1)
real_A: illumination patterns (batch_size * pixels * pixels * num_patterns)
batch_size: batch_size
img_W: width of image
img_H: high of image
num_A: num_patterns

Outputs:
out_x: estimated image by GIDC (batch_size * pixels * pixels * 1)
out_y: estimated intensity measurements associated with out_x and real_A (batch_size * 1 * 1 * num_patterns)
"""
# Unet
import tensorflow as tf

class GIDC_model_Unet(tf.keras.Model):
    def __init__(self):
        super(GIDC_model_Unet, self).__init__()
        c_size = 5
        d_size = 5
        leaky_relu = lambda x: tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        # Encoder
        self.conv0 = tf.keras.layers.Conv2D(16, c_size, padding='SAME', activation=leaky_relu)
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(16, c_size, padding='SAME', activation=leaky_relu)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1_1 = tf.keras.layers.Conv2D(16, c_size, padding='SAME', activation=leaky_relu)
        self.bn1_1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.Conv2D(16, d_size, strides=2, padding='SAME', activation=leaky_relu)
        self.bn_pool1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(32, c_size, padding='SAME', activation=leaky_relu)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2_1 = tf.keras.layers.Conv2D(32, c_size, padding='SAME', activation=leaky_relu)
        self.bn2_1 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.Conv2D(32, d_size, strides=2, padding='SAME', activation=leaky_relu)
        self.bn_pool2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(64, c_size, padding='SAME', activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv3_1 = tf.keras.layers.Conv2D(64, c_size, padding='SAME', activation=leaky_relu)
        self.bn3_1 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.Conv2D(64, d_size, strides=2, padding='SAME', activation=leaky_relu)
        self.bn_pool3 = tf.keras.layers.BatchNormalization()

        self.conv4 = tf.keras.layers.Conv2D(128, c_size, padding='SAME', activation=leaky_relu)
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.conv4_1 = tf.keras.layers.Conv2D(128, c_size, padding='SAME', activation=leaky_relu)
        self.bn4_1 = tf.keras.layers.BatchNormalization()
        self.pool4 = tf.keras.layers.Conv2D(128, d_size, strides=2, padding='SAME', activation=leaky_relu)
        self.bn_pool4 = tf.keras.layers.BatchNormalization()

        # Bottleneck
        self.conv5 = tf.keras.layers.Conv2D(256, c_size, padding='SAME', activation=leaky_relu)
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.conv5_1 = tf.keras.layers.Conv2D(256, c_size, padding='SAME', activation=leaky_relu)
        self.bn5_1 = tf.keras.layers.BatchNormalization()

        # Decoder
        self.up6 = tf.keras.layers.Conv2DTranspose(128, c_size, strides=2, padding='SAME', activation=leaky_relu)
        self.bn6 = tf.keras.layers.BatchNormalization()
        self.conv6_1 = tf.keras.layers.Conv2D(128, c_size, padding='SAME', activation=leaky_relu)
        self.bn6_1 = tf.keras.layers.BatchNormalization()
        self.conv6_2 = tf.keras.layers.Conv2D(128, c_size, padding='SAME', activation=leaky_relu)
        self.bn6_2 = tf.keras.layers.BatchNormalization()

        self.up7 = tf.keras.layers.Conv2DTranspose(64, c_size, strides=2, padding='SAME', activation=leaky_relu)
        self.bn7 = tf.keras.layers.BatchNormalization()
        self.conv7_1 = tf.keras.layers.Conv2D(64, c_size, padding='SAME', activation=leaky_relu)
        self.bn7_1 = tf.keras.layers.BatchNormalization()
        self.conv7_2 = tf.keras.layers.Conv2D(64, c_size, padding='SAME', activation=leaky_relu)
        self.bn7_2 = tf.keras.layers.BatchNormalization()

        self.up8 = tf.keras.layers.Conv2DTranspose(32, c_size, strides=2, padding='SAME', activation=leaky_relu)
        self.bn8 = tf.keras.layers.BatchNormalization()
        self.conv8_1 = tf.keras.layers.Conv2D(32, c_size, padding='SAME', activation=leaky_relu)
        self.bn8_1 = tf.keras.layers.BatchNormalization()
        self.conv8_2 = tf.keras.layers.Conv2D(32, c_size, padding='SAME', activation=leaky_relu)
        self.bn8_2 = tf.keras.layers.BatchNormalization()

        self.up9 = tf.keras.layers.Conv2DTranspose(16, c_size, strides=2, padding='SAME', activation=leaky_relu)
        self.bn9 = tf.keras.layers.BatchNormalization()
        self.conv9_1 = tf.keras.layers.Conv2D(16, c_size, padding='SAME', activation=leaky_relu)
        self.bn9_1 = tf.keras.layers.BatchNormalization()
        self.conv9_2 = tf.keras.layers.Conv2D(16, c_size, padding='SAME', activation=leaky_relu)
        self.bn9_2 = tf.keras.layers.BatchNormalization()

        self.conv10 = tf.keras.layers.Conv2D(1, c_size, padding='SAME', activation='sigmoid')
        self.bn10 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        inpt, real_A = inputs
        
        # Encoder
        c0 = self.bn0(self.conv0(inpt), training=training)
        c1 = self.bn1(self.conv1(c0), training=training)
        c1_1 = self.bn1_1(self.conv1_1(c1), training=training)
        p1 = self.bn_pool1(self.pool1(c1_1), training=training)

        c2 = self.bn2(self.conv2(p1), training=training)
        c2_1 = self.bn2_1(self.conv2_1(c2), training=training)
        p2 = self.bn_pool2(self.pool2(c2_1), training=training)

        c3 = self.bn3(self.conv3(p2), training=training)
        c3_1 = self.bn3_1(self.conv3_1(c3), training=training)
        p3 = self.bn_pool3(self.pool3(c3_1), training=training)

        c4 = self.bn4(self.conv4(p3), training=training)
        c4_1 = self.bn4_1(self.conv4_1(c4), training=training)
        p4 = self.bn_pool4(self.pool4(c4_1), training=training)

        # Bottleneck
        c5 = self.bn5(self.conv5(p4), training=training)
        c5_1 = self.bn5_1(self.conv5_1(c5), training=training)

        # Decoder
        u6 = self.bn6(self.up6(c5_1), training=training)
        merge6 = tf.concat([c4_1, u6], axis=3)
        c6_1 = self.bn6_1(self.conv6_1(merge6), training=training)
        c6_2 = self.bn6_2(self.conv6_2(c6_1), training=training)

        u7 = self.bn7(self.up7(c6_2), training=training)
        merge7 = tf.concat([c3_1, u7], axis=3)
        c7_1 = self.bn7_1(self.conv7_1(merge7), training=training)
        c7_2 = self.bn7_2(self.conv7_2(c7_1), training=training)

        u8 = self.bn8(self.up8(c7_2), training=training)
        merge8 = tf.concat([c2_1, u8], axis=3)
        c8_1 = self.bn8_1(self.conv8_1(merge8), training=training)
        c8_2 = self.bn8_2(self.conv8_2(c8_1), training=training)

        u9 = self.bn9(self.up9(c8_2), training=training)
        merge9 = tf.concat([c1_1, u9], axis=3)
        c9_1 = self.bn9_1(self.conv9_1(merge9), training=training)
        c9_2 = self.bn9_2(self.conv9_2(c9_1), training=training)

        out_x = self.bn10(self.conv10(c9_2), training=training)

        # the measurement process of ghost imaging (physical model)
        out_x_norm = out_x / tf.reduce_max(out_x)
                
        out_y = tf.reduce_sum(out_x_norm * real_A, axis=[1, 2], keepdims=True)
        out_y = tf.transpose(out_y, perm=[0, 2, 1, 3])
        
        # sometime the normalization helps
        mean_x, variance_x = tf.nn.moments(out_x, [0,1,2,3])
        mean_y, variance_y = tf.nn.moments(out_y, [0,1,2,3])
        out_x = (out_x - mean_x)/tf.sqrt(variance_x + 1e-8)
        out_y = (out_y - mean_y)/tf.sqrt(variance_y + 1e-8)

        return out_x,out_y








