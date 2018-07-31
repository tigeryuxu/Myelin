# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:53:34 2018

@author: Tiger
"""

""" Initializes cost function and weights
"""

import tensorflow as tf
from matplotlib import *
import numpy as np
import scipy
import math

from plot_functions import *
from data_functions import *
from post_process_functions import *
from UNet import *

""" SPATIAL WEIGHTING """
def spatial_weight(y_,edgeFalloff=10,background=0.01,approximate=True):
    if approximate:   # does chebyshev
        dist1 = scipy.ndimage.distance_transform_cdt(y_)
        dist2 = scipy.ndimage.distance_transform_cdt(numpy.where(y_>0,0,1))    # sets everything in the middle of the OBJECT to be 0
        
        
    else:   # does euclidean
        dist1 = scipy.ndimage.distance_transform_edt(y_, sampling=[1,1,1])
        dist2 = scipy.ndimage.distance_transform_edt(numpy.where(y_>0,0,1), sampling=[1,1,1])
    
    
    """ DO CLASS WEIGHTING instead of spatial weighting WITHIN the object """
    dist1[dist1 > 0] = 0.5
    
    
    dist = dist1+dist2
    attention = math.e**(1-dist/edgeFalloff) + background   # adds background so no loses go to zero
    attention /= numpy.average(attention)
    return numpy.reshape(attention,y_.shape)




def class_weight(y_, loss, weight=10.0):
     
    weight_mat = np.zeros(np.shape(y_))
    weight_mat[weight_mat == 0] = weight 
    weighted_loss = np.multiply(y_,weight_mat)          # multiply by label weights

    return weighted_loss 



""" Initialized cost function """
def costOptm(y, y_b, logits, weighted_labels, weight_mat=True):
    # Choose fitness/cost function. Many options:
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_b, logits=logits) # calculate loss
    
    #class_w_loss = class_weight(y_b, loss, class_1=1.0, class_2=10.0);   # PERFORMS CLASS WEIGHTING
    original = loss
    if weight_mat:
        w_reduced = tf.reduce_mean(weighted_labels, axis=-1)
        loss = tf.multiply(loss, w_reduced)
        
    cross_entropy = tf.reduce_mean(loss)         # single loss value
    
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)  # train_step uses Adam Optimizer
 
    """ Accuracy 
    """ 
    correct_prediction = tf.equal(tf.argmax(y, 3), tf.argmax(y_b, 3))  # accuracy prediction
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))    
    
    """ Jaccard
    """
    output = tf.cast(tf.argmax(y,axis=-1), dtype=tf.float32)
    truth = tf.cast(tf.argmax(y_b,axis=-1), dtype=tf.float32)
    intersection = tf.reduce_sum(tf.reduce_sum(tf.multiply(output, truth), axis=-1),axis=-1)
    union = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.add(output, truth)>= 1, dtype=tf.float32), axis=-1),axis=-1) + 0.0000001
    jaccard = tf.reduce_mean(intersection / union)   
    
    
    return accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original




""" Initialized cost function """
def costOptm_CLASSW(y, y_b, logits):
    # Choose fitness/cost function. Many options:
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_b, logits=logits) # calculate loss
    
    
    class_1 = 1.0
    class_2 = 10.0    
    shape = np.concatenate(([1], y_b.get_shape().as_list()[1:3], [1]), axis=0)
    first_c = tf.constant(class_1, shape=shape)
    second_c = tf.constant(class_2, shape=shape)
    weights = tf.concat([first_c, second_c], axis=-1)  
    multiplied = tf.multiply(y_b, weights)
    w_reduced = tf.reduce_mean(multiplied, axis=-1)
 
    weighted_loss = tf.multiply(loss, w_reduced)          # multiply by label weights
        
    cross_entropy = tf.reduce_mean(weighted_loss)         # single loss value
    
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)  # train_step uses Adam Optimizer
 
    """ Accuracy 
    """ 
    correct_prediction = tf.equal(tf.argmax(y, 3), tf.argmax(y_b, 3))  # accuracy prediction
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))    
    
    """ Jaccard
    """
    output = tf.cast(tf.argmax(y,axis=-1), dtype=tf.float32)
    truth = tf.cast(tf.argmax(y_b,axis=-1), dtype=tf.float32)
    intersection = tf.reduce_sum(tf.reduce_sum(tf.multiply(output, truth), axis=-1),axis=-1)
    union = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.add(output, truth)>= 1, dtype=tf.float32), axis=-1),axis=-1) + 0.0000001
    jaccard = tf.reduce_mean(intersection / union)   
    
    
    return accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, weighted_loss



""" No Weight """
def costOptm_noW(y, y_b, logits):
    # Choose fitness/cost function. Many options:
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_b, logits=logits) # calculate loss
    cross_entropy = tf.reduce_mean(loss)         # single loss value
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)  # train_step uses Adam Optimizer
 
    """ Accuracy 
    """ 
    correct_prediction = tf.equal(tf.argmax(y, 3), tf.argmax(y_b, 3))  # accuracy prediction
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))    
    
    """ Jaccard
    """
    output = tf.cast(tf.argmax(y,axis=-1), dtype=tf.float32)
    truth = tf.cast(tf.argmax(y_b,axis=-1), dtype=tf.float32)
    intersection = tf.reduce_sum(tf.reduce_sum(tf.multiply(output, truth), axis=-1),axis=-1)
    union = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.add(output, truth)>= 1, dtype=tf.float32), axis=-1),axis=-1) + 0.0000001
    jaccard = tf.reduce_mean(intersection / union)   
    
    
    weighted_loss = 0
    return accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, weighted_loss


""" Initialized cost function """
def costOptm_BOTH(y, y_b, logits, weighted_labels, weight_mat=True):
    # Choose fitness/cost function. Many options:
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_b, logits=logits) # calculate loss
    if weight_mat:
        w_reduced = tf.reduce_mean(weighted_labels, axis=-1)
        loss = tf.multiply(loss, w_reduced)    
    #loss = tf.cast(loss, tf.float64)
    
    
    class_1 = np.float32(1.0)
    class_2 = np.float32(10.0)   
    shape = np.concatenate(([1], y_b.get_shape().as_list()[1:3], [1]), axis=0) 
    first_c = tf.constant(class_1, shape=shape)
    second_c = tf.constant(class_2, shape=shape)
    weights = tf.concat([first_c, second_c], axis=-1)  
    multiplied = tf.multiply(y_b, weights)
    w_reduced = tf.reduce_mean(multiplied, axis=-1)
 
    weighted_loss = tf.multiply(loss, w_reduced)          # multiply by label weights
        
    cross_entropy = tf.reduce_mean(weighted_loss)         # single loss value
    
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)  # train_step uses Adam Optimizer
 
    """ Accuracy 
    """ 
    correct_prediction = tf.equal(tf.argmax(y, 3), tf.argmax(y_b, 3))  # accuracy prediction
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))    
    
    """ Jaccard
    """
    output = tf.cast(tf.argmax(y,axis=-1), dtype=tf.float32)
    truth = tf.cast(tf.argmax(y_b,axis=-1), dtype=tf.float32)
    intersection = tf.reduce_sum(tf.reduce_sum(tf.multiply(output, truth), axis=-1),axis=-1)
    union = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.add(output, truth)>= 1, dtype=tf.float32), axis=-1),axis=-1) + 0.0000001
    jaccard = tf.reduce_mean(intersection / union)   
        
    return accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, weighted_loss


""" Creates network architecture
"""
def create_network(x, y_b, training):
    # Building Convolutional layers
    siz_f = 5 # or try 5 x 5
    #training = True

    L1 = tf.layers.conv2d(inputs=x, filters=10, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_D')
    L2 = tf.layers.conv2d(inputs=L1, filters=20, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_D')
    L3 = tf.layers.conv2d(inputs=L2, filters=30, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_D')
    L4 = tf.layers.conv2d(inputs=L3, filters=40, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_D')
    L5 = tf.layers.conv2d(inputs=L4, filters=50, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_D')

    # up 1
    L6 = tf.layers.conv2d_transpose(inputs=L5, filters=50, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_D')
    L6_conv = tf.concat([L6, L4], axis=3)  # add earlier layers, then convolve together
    
    L7 = tf.layers.conv2d_transpose(inputs=L6_conv, filters=40, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_D')
    L7_conv = tf.concat([L7, L3], axis=3)

    L8 = tf.layers.conv2d_transpose(inputs=L7_conv, filters=30, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_D')
    L8_conv = tf.concat([L8, L2], axis=3)
     
    L9 = tf.layers.conv2d_transpose(inputs=L8_conv, filters=20, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_D')
    L9_conv = tf.concat([L9, L1], axis=3)

    L10 = tf.layers.conv2d_transpose(inputs=L9_conv, filters=10, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_D')
    L10_conv = tf.concat([L10, x], axis=3)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv2d(inputs=L10_conv, filters=2, kernel_size=[siz_f, siz_f], strides=1, padding='same', activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    
    return y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed