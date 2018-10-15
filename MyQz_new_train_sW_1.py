# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================


 ***NEED TO INSTALL SCIPY???
 
 
 Try reducing network size
 

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

from random import randint

from plot_functions import *
from data_functions import *
#from post_process_functions import *
from UNet import *
import glob, os



# Initialize everything with specific random seeds for repeatability
tf.set_random_seed(2);

"""  Network Begins:
"""
# for saving
s_path = 'C:/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Checkpoints/'
# for input
input_path = 'D:/Tiger/Tiger/AI stuff/z_myelin_data_FULL_with_NANOFIBERS_TRAINING/'



""" load mean and std """  
mean_arr = load_pkl('', 'mean_arr.pkl')
std_arr = load_pkl('', 'std_arr.pkl')
               
""" Load filenames from zip """
images = glob.glob(os.path.join(input_path,'*input.tif'))
examples = [dict(input=i,truth=i.replace('input.tif','truth.tif')) for i in images]

counter = list(range(len(examples)))  # create a counter, so can randomize it
counter = np.array(counter)
np.random.shuffle(counter)

val_size = 0.1;
val_idx_sub = round(len(counter) * val_size)
validation_counter = counter[-1 - val_idx_sub : -1]
input_counter = counter[0: -1 - val_idx_sub]

# Variable Declaration
x = tf.placeholder('float32', shape=[None, 1024, 1024, 3], name='InputImage')
y_ = tf.placeholder('float32', shape=[None, 1024, 1024, 2], name='CorrectLabel')
training = tf.placeholder(tf.bool, name='training')
weight_matrix = tf.placeholder('float32', shape=[None, 1024, 1024, 2], name = 'weighted_labels')


""" Creates network and cost function"""
y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed = create_network(x, y_, training)
accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm(y, y_b, logits, weight_matrix, weight_mat=True)

sess = tf.InteractiveSession()

""" TO LOAD OLD CHECKPOINT """
saver = tf.train.Saver()

# Read in file names
onlyfiles_check = [ f for f in listdir(s_path) if isfile(join(s_path,f))]   
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
onlyfiles_check.sort(key = natsort_key1)

""" If no old checkpoint then starts fresh """
if len(onlyfiles_check) < 8:  
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run() 
    plot_cost = []; plot_cost_val = []; plot_jaccard = []; plot_jaccard_val = [];
    num_check= 0;

else:   
    """ Find last checkpoint """   
    last_file = onlyfiles_check[-8]
    split = last_file.split('.')
    checkpoint = split[0]
    num_check = checkpoint.split('_')
    num_check = int(num_check[1])
    
    saver.restore(sess, s_path + checkpoint)
    
    # Getting back the objects:
    with open(s_path + 'loss_global.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        plot_cost = loaded[0]
    
    # Getting back the objects:
    with open(s_path + 'loss_global_val.pkl', 'rb') as t:  # Python 3: open(..., 'rb')
        loaded = pickle.load(t)
        plot_cost_val = loaded[0]  
    
    # Getting back the objects:
    with open(s_path + 'jaccard.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        plot_jaccard = loaded[0] 

    # Getting back the objects:
    with open(s_path + 'jaccard_val.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        plot_jaccard_val = loaded[0] 
        
        
    # Getting back the objects:
    with open(s_path + 'val_counter.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        validation_counter = loaded[0]     
        
    # Getting back the objects:
    with open(s_path + 'input_counter.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        input_counter = loaded[0]  
  


# Required to initialize all
batch_size = 4; 
save_epoch = 1000;
plot_every = 10;
epochs = num_check;

batch_x = []; batch_y = [];
weights = [];

for P in range(8000000000000000000000):
    np.random.shuffle(validation_counter)
    np.random.shuffle(input_counter)
    for i in range(len(input_counter)):
        
        # degrees rotated:
        rand = randint(0, 360)      
        """ Load input image """
        input_name = examples[input_counter[i]]['input']
        input_im = np.asarray(Image.open(input_name), dtype=np.float32)

        # ROTATE the input_im
        np_zeros = np.zeros([1024, 1024, 3])
        np_zeros[:,192:832, :] = input_im[:, :, :]
        
        im = Image.fromarray(np.asarray(np_zeros, dtype=np.uint8))
        rotated = im.rotate(rand)
        input_im = np.asarray(rotated, dtype=np.float32)


        """ Load truth image """
        truth_name = examples[input_counter[i]]['truth']
        truth_tmp = np.asarray(Image.open(truth_name), dtype=np.float32)

        # ROTATE the truth_im
        np_zeros = np.zeros([1024, 1024])
        np_zeros[:,192:832] = truth_tmp[:, :]
        im = Image.fromarray(np.asarray(np_zeros, dtype=np.uint8))
        rotated = im.rotate(rand)
        truth_tmp = np.asarray(rotated, dtype=np.float32)


        """ maybe remove normalization??? """
        input_im = normalize_im(input_im, mean_arr, std_arr) 
        
        """ convert truth to 2 channel image """
        channel_1 = np.copy(truth_tmp)
        channel_1[channel_1 == 0] = 1
        channel_1[channel_1 == 255] = 0
                
        channel_2 = np.copy(truth_tmp)
        channel_2[channel_2 == 255] = 1   
        
        truth_im = np.zeros(np.shape(truth_tmp) + (2,))
        truth_im[:, :, 0] = channel_1   # background
        truth_im[:, :, 1] = channel_2   # blebs
            
        blebs_label = np.copy(truth_im[:, :, 1])
        
        """ Get spatial AND class weighting mask for truth_im """
        sp_weighted_labels = spatial_weight(blebs_label,edgeFalloff=10,background=0.01,approximate=True)
        
        """ OR DO class weighting ONLY """
        #c_weighted_labels = class_weight(blebs_label, loss, weight=10.0)        
        
        """ Create a matrix of weighted labels """
        weighted_labels = np.copy(truth_im)
        weighted_labels[:, :, 1] = sp_weighted_labels
        
        """ set inputs and truth """
        batch_x.append(input_im)
        batch_y.append(truth_im)
        weights.append(weighted_labels)
                
        """ Plot for debug """
#        plt.figure(1); 
#        plt.subplot(221); plt.imshow(np.asarray(input_im, dtype = np.uint8)); plt.title('Input');
#        plt.subplot(222); plt.imshow(sp_weighted_labels); plt.title('weighted');    plt.pause(0.005)
#        plt.subplot(223); plt.imshow(channel_1); plt.title('background');
#        plt.subplot(224); plt.imshow(channel_2); plt.title('blebs');
    
        """ Feed into training loop """
        if len(batch_x) == batch_size:
           feed_dict_TRAIN = {x:batch_x, y_:batch_y, training:1, weight_matrix:weights}                 

           train_step.run(feed_dict=feed_dict_TRAIN)

           batch_x = []; batch_y = []; weights = [];
           epochs = epochs + 1           
           print('Trained: %d' %(epochs))
           
           
           if epochs % plot_every == 0:
              plt.close(2)
              batch_x_val = []
              batch_y_val = []
              batch_weights_val = []
              for batch_i in range(len(validation_counter)):

                  # degrees rotated:
                  rand = randint(0, 360)  
                  
                  # select random validation image:
                  rand_idx = randint(0, len(validation_counter)- 1)
                          
                  """ Load input image """
                  input_name = examples[validation_counter[rand_idx]]['input']
                  input_im_val = np.asarray(Image.open(input_name), dtype=np.float32)
            
                  # ROTATE the input_im
                  np_zeros = np.zeros([1024, 1024, 3])
                  np_zeros[:,192:832, :] = input_im_val[:, :, :]
                    
                  im = Image.fromarray(np.asarray(np_zeros, dtype=np.uint8))
                  rotated = im.rotate(rand)
                  input_im_val = np.asarray(rotated, dtype=np.float32)
            
                  """ Load truth image """
                  truth_name = examples[validation_counter[rand_idx]]['truth']
                  truth_tmp_val = np.asarray(Image.open(truth_name), dtype=np.float32)
            
                  # ROTATE the truth_im
                  np_zeros = np.zeros([1024, 1024])
                  np_zeros[:,192:832] = truth_tmp_val[:, :]
                  im = Image.fromarray(np.asarray(np_zeros, dtype=np.uint8))
                  rotated = im.rotate(rand)
                  truth_tmp_val = np.asarray(rotated, dtype=np.float32)

            
                  """ maybe remove normalization??? """
                  input_im_val = normalize_im(input_im_val, mean_arr, std_arr) 
                
                  """ convert truth to 2 channel image """
                  channel_1 = np.copy(truth_tmp_val)
                  channel_1[channel_1 == 0] = 1
                  channel_1[channel_1 == 255] = 0
                            
                  channel_2 = np.copy(truth_tmp_val)
                  channel_2[channel_2 == 255] = 1   
                      
                  truth_im_val = np.zeros(np.shape(truth_tmp_val) + (2,))
                  truth_im_val[:, :, 0] = channel_1   # background
                  truth_im_val[:, :, 1] = channel_2   # blebs
        
                  blebs_label = np.copy(truth_im_val[:, :, 1])
                 
                  """ Get spatial AND class weighting mask for truth_im """
                  sp_weighted_labels = spatial_weight(blebs_label,edgeFalloff=10,background=0.01,approximate=True)
                 
                                 
                  """ OR DO class weighting ONLY """
                  #c_weighted_labels = class_weight(blebs_label, loss, weight=10.0)        
                
                  """ Create a matrix of weighted labels """
                  weighted_labels_val = np.copy(truth_im_val)
                  weighted_labels_val[:, :, 1] = sp_weighted_labels
                
                  """ set inputs and truth """
                  batch_x_val.append(input_im_val)
                  batch_y_val.append(truth_im_val)
                  batch_weights_val.append(weighted_labels_val)
                  
                  if len(batch_x_val) == batch_size:
                      break

             
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
#              plot_cost_fun(plot_cost, plot_cost_val)
#              plot_jaccard_fun(plot_jaccard, plot_jaccard_val)
#             
#                
#              """ Plot for debug """
#              batch_x.append(input_im); batch_y.append(truth_im); weights.append(weighted_labels);
#              feed_dict = {x:batch_x, y_:batch_y, training:1, weight_matrix:weights}  
#              output_train = softMaxed.eval(feed_dict=feed_dict)
#              seg_train = np.argmax(output_train, axis = -1)[0]              
#              
#              batch_x = []; batch_y = []; weights = [];
#              batch_x.append(input_im_val); batch_y.append(truth_im_val); weights.append(weighted_labels_val);
#              feed_dict = {x:batch_x, y_:batch_y, training:1, weight_matrix:weights}  
#              output_val = softMaxed.eval(feed_dict=feed_dict)
#              seg_val = np.argmax(output_val, axis = -1)[0]    
#              
#              plt.figure(2);
#              plt.subplot(221); plt.imshow(truth_tmp); plt.title('Truth Train');
#              plt.subplot(222); plt.imshow(seg_train); plt.title('Output Train');              
#              plt.subplot(223); plt.imshow(truth_tmp_val); plt.title('Truth Validation');        
#              plt.subplot(224); plt.imshow(seg_val); plt.title('Output Validation'); plt.pause(0.0005);
#              
#              
#              if epochs > 500:
#                  if epochs % 500 == 0:
#                      plt.savefig(s_path + '_' + str(epochs) + '_output.png')
#              elif epochs % 10 == 0:
#                  plt.savefig(s_path + '_' + str(epochs) + '_output.png')
#              
#              batch_x = []; batch_y = []; weights = [];
#              
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
              save_pkl(validation_counter, s_path, 'val_counter.pkl')
              save_pkl(input_counter, s_path, 'input_counter.pkl')   
                                            
              """Getting back the objects"""
  #           plot_cost = load_pkl(s_path, 'loss_global.pkl')
  #           plot_cost_val = load_pkl(s_path, 'loss_global_val.pkl')
  #           plot_jaccard = load_pkl(s_path, 'jaccard.pkl')
  #           plot_jaccard = load_pkl(s_path, 'jaccard_va;.pkl')