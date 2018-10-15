# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:54:37 2018

@author: Tiger
"""


import tensorflow as tf
from matplotlib import *
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
from skimage import measure
import pickle
import os
from skimage.filters import threshold_mean

from plot_functions import *
from data_functions import *
from post_process_functions import *
from UNet import *
from pre_processing import *



#s_path = 'C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Checkpoints/Check_MyQz_new_train_sW_1_rotated/'    
#sess = tf.InteractiveSession()
#    
#plot_cost_val = []
#plot_jaccard = []
#for i in range(1000, 301 * 1000, 1000):
#        """ TO LOAD OLD CHECKPOINT """
#        saver = tf.train.Saver()
#        saver.restore(sess, s_path + 'check_' + str(i))
#                      
#        """ Training loss"""
#        loss_t = cross_entropy.eval(feed_dict=feed_dict)
#        plot_cost_val.append(loss_t)
#        
#        """ Training loss"""
#        jaccard_t = jaccard.eval(feed_dict=feed_dict)
#        plot_jaccard.append(jaccard_t)
#
#
#""" function call to plot """
#plot_cost_fun([], plot_cost_val)
#plot_jaccard_fun(plot_jaccard, [])               
#    
#
#
#    
#def batch_4():
    
    
rotate = 1
batch_size=2

len_x = 1024
width_x = 640  # change to 1024 for rotate
if rotate:
    width_x = 1024


#s_path = 'D:/Tiger/AI stuff/MyelinUNet/Checkpoints/Check_MyQz_new_train_sW_1_rotated/' 
s_path = 'D:/Tiger/AI stuff/MyelinUNet/Checkpoints/Check_MyQz_new_train_sW_1_with_diff_counter/' 

#s_path = 'D:/Tiger/AI stuff/MyelinUNet/Checkpoints/Check_MyQz_new_train_sW_2_not_rotated/'  
x = tf.placeholder('float32', shape=[None, len_x, width_x, 3], name='InputImage')
y_ = tf.placeholder('float32', shape=[None, len_x, width_x, 2], name='CorrectLabel')
training = tf.placeholder(tf.bool, name='training')
weight_matrix = tf.placeholder('float32', shape=[None, len_x, width_x, 2], name = 'weighted_labels')

y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed = create_network(x, y_, training)
accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm(y, y_b, logits, weight_matrix, weight_mat=True)

# for validation
val_input_path='D:/Tiger/AI stuff/MyelinUNet/Validation/Tmp/'

#input_path = 'E:/Tiger/UNet_new_data_1000px/'

""" Load avg_img and std_img from TRAINING SET """
mean_arr = 0; std_arr = 0;
# Getting back the objects:
with open('mean_arr.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    loaded = pickle.load(f)
    mean_arr = loaded[0]
# Getting back the objects:
with open('std_arr.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    loaded = pickle.load(f)
    std_arr = loaded[0]
    
myzip_val, onlyfiles_val, counter_val = read_zip_names(val_input_path, 'validation_13.zip')
   
""" parse validation into counter WITH fibers, and those with NO fibers """
counter_fibers, counter_blank = parse_validation(myzip_val, onlyfiles_val, counter_val)
 
sess = tf.InteractiveSession()

plot_cost_val = []
plot_jaccard = []
for i in range(500 * 1000, 990 * 1000, 1000):
    
    """ TO LOAD OLD CHECKPOINT """
    saver = tf.train.Saver()
    saver.restore(sess, s_path + 'check_' + str(i))
    

    """ GET VALIDATION """
    batch_x_val_fibers, batch_y_val_fibers, batch_weights_fibers = get_batch_val(myzip_val, onlyfiles_val, counter_fibers, mean_arr, std_arr, 
                                                           batch_size=batch_size/2, rotate=rotate)
    #batch_x_val_empty, batch_y_val_empty, batch_weights_empty = get_batch_val(myzip_val, onlyfiles_val, counter_blank, mean_arr, std_arr,
    #                                                     batch_size=batch_size/2)
    #batch_y_val = batch_y_val_fibers + batch_y_val_empty
    #batch_x_val = batch_x_val_fibers + batch_x_val_empty
    #batch_weights_val = batch_weights_fibers + batch_weights_empty
    
    batch_y_val = batch_y_val_fibers
    batch_x_val = batch_x_val_fibers
    
    batch_weights_val = batch_weights_fibers
             
    feed_dict = {x:batch_x_val, y_:batch_y_val, training:0, weight_matrix:batch_weights_val}    
    
    #feed_dict = {x:batch_x_val, y_:batch_y_val, training:0}
    """ Validation loss """
    loss_t = cross_entropy.eval(feed_dict=feed_dict)
    plot_cost_val.append(loss_t)
    
    """ Validation jaccard """
    jaccard_t = jaccard.eval(feed_dict=feed_dict)
    plot_jaccard.append(jaccard_t)


""" function call to plot """
plot_cost_fun([], plot_cost_val)
plot_jaccard_fun(plot_jaccard, [])    
   

""" Saving the objects """
save_pkl(plot_cost_val, s_path, 'loss_global_MyQz9_noW.pkl')
save_pkl(plot_jaccard, s_path, 'jaccard_MyQ9_noW.pkl')
