# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:24:04 2018

@author: Neuroimmunology Unit
"""

# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================

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
import pickle as pickle
import os
import zipfile
import scipy

from plot_functions import *
from data_functions import *
#from post_process_functions import *
from UNet import *


"""  Network Begins:
"""
# for saving
s_path = '/project/6015947/yxu233/MyelinUNet_new/Checkpoints/Check_MyQz16_QL_BOTHW/'
# for input
input_path = '/scratch/yxu233/'
# for validation
val_input_path='/project/6015947/yxu233/MyelinUNet_new/create_training/Output/QL/New_Validation/'

# for saving
s_path = 'C:/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Checkpoints/Check_MyQz16_QL_BOTHW_by_3/'
# for input
input_path = 'D:/Tiger/AI stuff/OUTPUT/'
# for validation
val_input_path='D:/Tiger/AI stuff/Validation/NOT_AUGUST/'


""" load mean and std """  
mean_arr = load_pkl('', 'mean_arr.pkl')
std_arr = load_pkl('', 'std_arr.pkl')

""" Load filenames from zip """
onlyfiles_mask = read_file_names(input_path)
onlyfiles_val = read_file_names(val_input_path)

""" parse validation into counter WITH fibers, and those with NO fibers """
counter_fibers = []
counter_blank = []
for T in range(len(onlyfiles_val)):
    """ Get validation images """
    filename = onlyfiles_val[T]
    
    input_val, truth_val = load_training_bz(val_input_path, filename)
    
    """ Check to see if contains fibers or not """
    if np.count_nonzero(truth_val[:, :, 1]) > 0:     # IF YES, there are fibers
        counter_fibers.append(T)

    elif np.count_nonzero(truth_val[:, :, 1]) == 0:
        counter_blank.append(T)

# Variable Declaration

# 1460, 1936 doesn't work
x = tf.placeholder('float32', shape=[None, 1440, 1920, 3], name='InputImage')
y_ = tf.placeholder('float32', shape=[None, 1440, 1920, 2], name='CorrectLabel')
training = tf.placeholder(tf.bool, name='training')
weight_matrix = tf.placeholder('float32', shape=[None, 1440, 1920, 2], name = 'weighted_labels')

""" Creates network and cost function"""
y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed = create_network(x, y_, training)
accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm_BOTH(y, y_b, logits, weight_matrix, weight_mat=True)

sess = tf.InteractiveSession()

# Required to initialize all
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

batch_size = 2; 
batch_size_val = 2;
save_epoch = 1000;
plot_cost = []; plot_cost_val = []; plot_jaccard = []; plot_jaccard_val = [];
epochs = 0;

batch_x = []; batch_y = [];
weights = [];

for P in range(8000000000000000000000):
    counter = list(range(len(onlyfiles_mask)))  # create a counter, so can randomize it
    counter = np.array(counter)
    np.random.shuffle(counter)
    
    for i in range(len(onlyfiles_mask)):
        
        filename = onlyfiles_mask[counter[i]]
        input_im, truth_im = load_training_bz(input_path, filename)
        
#        input_im = input_im[0:1440, 0:1920, :]
#        truth_im = truth_im[0:1440, 0:1920, :]
            
        if input_im.shape[1] > 1500:
            input_im, truth_im = check_shape_QL(input_im, truth_im, len_im=1024, width_im=640)
        

        
        if not input_im.size:   # ERROR CATCHING
          print("EOF-error")
          continue
    
        """ Normalize the image first """
        input_crop = normalize_im(input_im, mean_arr, std_arr)  
           
        fiber_label = np.copy(truth_im[:, :, 1])
        
               
        """ Get spatial AND class weighting mask for truth_im """
        sp_weighted_labels = spatial_weight(fiber_label,edgeFalloff=10,background=0.01,approximate=True)
        
        """ OR DO class weighting ONLY """
        #c_weighted_labels = class_weight(fiber_label, loss, weight=10.0)        
        
        """ Create a matrix of weighted labels """
        weighted_labels = np.copy(truth_im)
        weighted_labels[:, :, 1] = sp_weighted_labels
        

        weighted_labels = weighted_labels.astype('float32')
        """ set inputs and truth """
        batch_x.append(input_crop)
        batch_y.append(truth_im)
        weights.append(weighted_labels)
                

        """ Plot for debug """
        plt.figure('Input'); plt.clf(); show_norm(batch_x[0]); plt.pause(0.05); 
        plt.figure('Truth'); plt.clf(); 
        true_m = np.argmax((batch_y[0]).astype('uint8'), axis=-1); plt.imshow(true_m);
        plt.pause(0.05); 
    
        """ Feed into training loop """
        if len(batch_x) == batch_size:
           feed_dict_TRAIN = {x:batch_x, y_:batch_y, training:1, weight_matrix:weights}                                  

           train_step.run(feed_dict=feed_dict_TRAIN)

           batch_x = []; batch_y = []; weights = [];
                     
           print('Trained: %d' %(epochs))
           
           
           if epochs % 10 == 0:
    
              """ GET VALIDATION """
              batch_x_val_fibers, batch_y_val_fibers, batch_weights_fibers = get_batch_val_bz(val_input_path, onlyfiles_val, counter_fibers, mean_arr, std_arr, 
                                                           batch_size=batch_size_val)
              batch_y_val = batch_y_val_fibers
              batch_x_val = batch_x_val_fibers
              batch_weights_val = batch_weights_fibers
             
              feed_dict_CROSSVAL = {x:batch_x_val, y_:batch_y_val, training:0, weight_matrix:batch_weights_val}         
              
              """ Training loss"""
              loss_t = cross_entropy.eval(feed_dict=feed_dict_TRAIN);
              plot_cost.append(loss_t);                 
                
              """ Training Jaccard """
              jacc_t = jaccard.eval(feed_dict=feed_dict_TRAIN)
              plot_jaccard.append(jacc_t)           
              
              """ CV loss """
              loss_val = cross_entropy.eval(feed_dict=feed_dict_CROSSVAL)
              plot_cost_val.append(loss_val)
             
              """ CV Jaccard """
              jacc_val = jaccard.eval(feed_dict=feed_dict_CROSSVAL)
              plot_jaccard_val.append(jacc_val)
              
              """ function call to plot """
              plot_cost_fun(plot_cost, plot_cost_val)
              plot_jaccard_fun(plot_jaccard, plot_jaccard_val)
        
           """ To save (every x epochs) """
           if epochs % save_epoch == 0:                          
              sess_get = tf.get_default_session()   # IF FORGET TO ADD "sess = tf.InteractiveSession
              saver = tf.train.Saver()
       
              save_name = s_path + 'check_' +  str(epochs)
              save_path = saver.save(sess_get, save_name)
               
              """ Saving the objects """
              save_pkl(plot_cost, s_path, 'loss_global.pkl')
              save_pkl(plot_cost_val, s_path, 'loss_global_val.pkl')
              save_pkl(plot_jaccard, s_path, 'jaccard.pkl')
              save_pkl(plot_jaccard_val, s_path, 'jaccard_val.pkl')
                                                             
              """Getting back the objects"""
              plot_cost = load_pkl(s_path, 'loss_global.pkl')
              plot_cost_val = load_pkl(s_path, 'loss_global_val.pkl')
              plot_jaccard = load_pkl(s_path, 'jaccard.pkl')
              plot_jaccard_val = load_pkl(s_path, 'jaccard_val.pkl')
  
           epochs = epochs + 1 