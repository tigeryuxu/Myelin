# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================


 ***NEED TO INSTALL SCIPY???

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
s_path = '/project/6015947/yxu233/MyelinUNet_new/Checkpoints/Check_MyQz10_all_sizes/'

# for input
input_path = '/project/6015947/yxu233/MyelinUNet_new/create_training/Output/'
# for validation
val_input_path='/project/6015947/yxu233/MyelinUNet_new/Validation/'


"""  Network Begins:
"""
# for saving
s_path = 'C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Checkpoints/Check_MyQ6/'

# for validation
val_input_path='C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Validation/'

input_path = 'E:/Tiger/UNet_new_data_1000px/'


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
    
""" Load validation """
# Read in file names
onlyfiles_val = [ f for f in listdir(val_input_path) if isfile(join(val_input_path,f))]   
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
onlyfiles_val.sort(key = natsort_key1)

batch_x_val = [];
batch_y_val = [];
weights_val = [];
for T in range(len(onlyfiles_val)):
    """ Get validation images """
    input_val, truth_val = load_training(val_input_path, onlyfiles_val[T])
    """ CONCATENATE ANOTHER MATRIX if < 3 channels """
    if np.shape(input_val)[-1] < 4:
          DAPI_val = np.copy(input_val[:, :, 1])
          input_val[:,:,1] = np.zeros([1024,640])
          DAPI_val = np.expand_dims(DAPI_val, axis=-1)
          input_val = np.append(input_val, DAPI_val, axis=-1)
            
    """ Normalize the image first """
    input_val = normalize_im(input_val, mean_arr, std_arr) 
    
    fiber_label = truth_val[:, :, 1]
        
    """ Get spatial AND class weighting mask for truth_im """
    sp_weighted_labels = spatial_weight(fiber_label,edgeFalloff=10,background=0.01,approximate=True)
        
    """ OR DO class weighting ONLY """
    #c_weighted_labels = class_weight(fiber_label, loss, weight=10.0)        
    
    """ Create a matrix of weighted labels """
    weighted_labels = np.copy(truth_val)
    weighted_labels[:, :, 1] = sp_weighted_labels

    """ set inputs and truth """
    batch_x_val.append(input_val)
    batch_y_val.append(truth_val)
    weights_val.append(weighted_labels)
    

# Variable Declaration
x = tf.placeholder('float32', shape=[None, 1024, 640, 4], name='InputImage')
y_ = tf.placeholder('float32', shape=[None, 1024, 640, 2], name='CorrectLabel')
training = tf.placeholder(tf.bool, name='training')
weight_matrix = tf.placeholder('float32', shape=[None, 1024, 640, 2], name = 'weighted_labels')

# Read in file names
myzip = zipfile.ZipFile(input_path + 'TEST_all_sizes.zip', 'r')
onlyfiles_mask = myzip.namelist()
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
onlyfiles_mask.sort(key = natsort_key1)

counter = list(range(len(onlyfiles_mask)))  # create a counter, so can randomize it
 
""" Creates network and cost function"""
y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed = create_network(x, y_, training)
accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm_CLASSW(y, y_b, logits)

sess = tf.InteractiveSession()
# Required to initialize all
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()
    
###
batch_size = 2; 
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
        input_im, truth_im = load_training_ZIP(myzip, filename)
        if not input_im.size:   # ERROR CATCHING
          print("EOF-error")
          continue
    
        """ CONCATENATE ANOTHER MATRIX if < 3 channels """
        if np.shape(input_im)[-1] < 4:
            DAPI_im = np.copy(input_im[:, :, 1])
            input_im[:,:,1] = np.zeros([1024,640])
            DAPI_im = np.expand_dims(DAPI_im, axis=-1)
            input_im = np.append(input_im, DAPI_im, axis=-1)
            

        """ Normalize the image first """
        input_crop = normalize_im(input_im, mean_arr, std_arr)  
           
        #""" set inputs and truth """
        #batch_x.append(input_crop)
        #batch_y.append(truth_im)
        
        fiber_label = truth_im[:, :, 1]
        
        """ Get spatial AND class weighting mask for truth_im """
        sp_weighted_labels = spatial_weight(fiber_label,edgeFalloff=10,background=0.01,approximate=True)
        
        """ OR DO class weighting ONLY """
        #c_weighted_labels = class_weight(fiber_label, loss, weight=10.0)        
        
        """ Create a matrix of weighted labels """
        weighted_labels = np.copy(truth_im)
        weighted_labels[:, :, 1] = sp_weighted_labels
    
        """ set inputs and truth """
        batch_x.append(input_crop)
        batch_y.append(truth_im)
        weights.append(weighted_labels)
    
        """ Plot for debug """
#        plt.figure('Input'); plt.clf(); show_norm(batch_x[0]); plt.pause(0.05); 
#        plt.figure('Truth'); plt.clf(); 
#        true_m = np.argmax((batch_y[0]).astype('uint8'), axis=-1); plt.imshow(true_m);
#        plt.pause(0.05); 

        print(epochs)
    
        """ Feed into training loop """
        if len(batch_x) == batch_size:
           feed_dict_TRAIN = {x:batch_x, y_:batch_y, training:1, weight_matrix:weights}                  

           train_step.run(feed_dict=feed_dict_TRAIN)
           
           batch_x = []; batch_y = [] 
           epochs = epochs + 1
           if epochs % 5 == 0:
              feed_dict_VAL = {x:batch_x_val, y_:batch_y_val, training:0, weight_matrix:weights_val}      
                        
              """ Training loss"""
              loss_t = cross_entropy.eval(feed_dict=feed_dict_TRAIN);
              plot_cost.append(loss_t);                 
                
              """ Training Jaccard """
              jacc_t = jaccard.eval(feed_dict=feed_dict_TRAIN)
              plot_jaccard.append(jacc_t)           
              
              """ CV loss """
              loss_val = cross_entropy.eval(feed_dict=feed_dict_VAL)
              plot_cost_val.append(loss_val)
             
              """ CV Jaccard """
              jacc_val = jaccard.eval(feed_dict=feed_dict_VAL)
              plot_jaccard_val.append(jacc_val)
              
              """ function call to plot """
              plot_cost_fun(plot_cost, plot_cost_val)
              plot_jaccard_fun(plot_jaccard, plot_jaccard_val)
        
           """ To save (every 1000 epochs) """
           if epochs % save_epoch == 0:                          
              sess_get = tf.get_default_session()   # IF FORGET TO ADD "sess = tf.InteractiveSession
              saver = tf.train.Saver()
       
              save_name = s_path + 'check_' +  str(epochs)
              save_path = saver.save(sess_get, save_name)
               
              """ Saving the objects """
              save_pkl(plot_cost, s_path, 'loss_global.pkl')
              save_pkl(plot_cost_val, s_path, 'loss_global_val.pkl')
              save_pkl(plot_jaccard, s_path, 'jaccard.pkl')
#              save_pkl(plot_jaccard_val, s_path, 'jaccard_val.pkl')
                                                             
              """Getting back the objects"""
#             plot_cost = load_pkl(s_path, 'loss_global.pkl')
#             plot_cost_val = load_pkl(s_path, 'loss_global_val.pkl')
#             plot_jaccard = load_pkl(s_path, 'jaccard.pkl')
#             plot_jaccard = load_pkl(s_path, 'jaccard_va;.pkl')
               