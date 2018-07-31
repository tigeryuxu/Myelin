# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:24:11 2017

@author: Tiger
"""

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


from plot_functions import *
from data_functions import *


# for saving
s_path = 'C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Checkpoints/'

# for training
input_path='C:/Users/Tiger/Documents/Tiger 2015/Antel Lab/AI_data/10x_full_well_masks_and_input/Input/'
DAPI_path='C:/Users/Tiger/Documents/Tiger 2015/Antel Lab/AI_data/10x_full_well_masks_and_input/DAPI_masks/'
mask_path='C:/Users/Tiger/Documents/Tiger 2015/Antel Lab/AI_data/10x_full_well_masks_and_input/Fibers_masks/'

# for validation
#input_path='C:/Users/Tiger/Documents/Tiger 2015/Antel Lab/AI_data/10x_validation/Input/'
#DAPI_path='C:/Users/Tiger/Documents/Tiger 2015/Antel Lab/AI_data/10x_validation/DAPI_masks/'
#mask_path='C:/Users/Tiger/Documents/Tiger 2015/Antel Lab/AI_data/10x_validation/Fibers_masks/'

# Read in file names
onlyfiles_mask = [ f for f in listdir(mask_path) if isfile(join(mask_path,f))]   
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
onlyfiles_mask.sort(key = natsort_key1)

# Read in file names
onlyfiles_DAPI = [ f for f in listdir(DAPI_path) if isfile(join(DAPI_path,f))]
onlyfiles_DAPI.sort(key = natsort_key1)
  
# Read in truth image names
onlyfiles_test = [ f for f in listdir(input_path) if isfile(join(input_path,f))] 
onlyfiles_test.sort(key = natsort_key1)

counter = list(range(len(onlyfiles_test)))  # create a counter, so can randomize it

###
batch_size = 1; 
plot_cost = []; plot_cost_val = [];
plot_acc = [];
num_train = 0;
total_counter = 1

""" Read in files all at once and save them"""
i = 0
while i < len(counter): #length of your filename      
      input_arr = readIm_counter(input_path,onlyfiles_test, counter[i])  
      DAPI_arr = readIm_counter(DAPI_path,onlyfiles_DAPI, counter[i])  
      mask_arr = readIm_counter(mask_path,onlyfiles_mask, counter[i])  
    
      DAPI_tmp = np.asarray(DAPI_arr, dtype=float)
      labelled = measure.label(DAPI_tmp)
      cc = measure.regionprops(labelled)
      
      # SHOULD RANDOMIZE THE COUNTER      
      counter_DAPI = list(range(len(cc)))  # create a counter, so can randomize it
      counter_DAPI = np.array(counter_DAPI)
      #np.random.shuffle(counter_DAPI)
      
      while N < len(cc):  
          DAPI_idx = cc[counter_DAPI[N]]['centroid']
          
          # extract CROP outo of everything          
          DAPI_crop, coords = adapt_crop_DAPI(DAPI_arr, DAPI_idx, length=1024, width=640)                    
          truth_crop, coords = adapt_crop_DAPI(mask_arr, DAPI_idx, length=1024, width=640)
          input_crop, coords = adapt_crop_DAPI(input_arr, DAPI_idx, length=1024, width=640)         
          
          """ Find fibers (truth_mask should already NOT contain DAPI, so don't need to get rid of it)
              ***however, the DAPI pixel value of DAPI_center should be the SAME as fibers pixel value + 1
          """
          val_at_center = DAPI_tmp[DAPI_idx[0].astype(int), DAPI_idx[1].astype(int)] 
          val_fibers = val_at_center + 1
          

          # Find all the ones that are == val_fibers
          truth_crop[truth_crop != val_fibers] = 0
          truth_crop[truth_crop == val_fibers] = 1
          
          # then split into binary classifier truth:
          fibers = np.copy(truth_crop)
          fibers = np.expand_dims(fibers, axis=3)
          
          null_space = np.copy(truth_crop)
          null_space[null_space == 0] = -1
          null_space[null_space > -1] = 0
          null_space[null_space == -1] = 1
          null_space = np.expand_dims(null_space, axis=3)
          
          combined = np.append(null_space, fibers, -1)
          
          DAPI_crop[DAPI_crop != val_at_center] = 0
          DAPI_crop[DAPI_crop == val_at_center] = 255
           
          """ Delete green channel by making it the DAPI_mask instead """
          input_crop[:, :, 1] = DAPI_crop
          
          concate_input = np.zeros([1024, 640, 5])
          
          concate_input[:, :, 0:3] = input_crop
          concate_input[:, :, 3:5] = combined
          
                      
          plt.figure('Batched'); plt.clf()
          plt.subplot(211); plt.imshow(concate_input[:, :, 0:3]/255);
          true_m = np.argmax((concate_input[:, :, 3:5]).astype('uint8'), axis=-1)  
          plt.subplot(212); plt.imshow(true_m, vmin = 0, vmax = 1);  plt.pause(0.05)                

          """ Saves image """        
          # Saving the objects:
          filename = onlyfiles_test[counter[i]]
          split_n = filename.split('.')
          with open(s_path + split_n[0] + '_' + str(N), 'wb') as f:  # Python 3: open(..., 'wb')
               pickle.dump([concate_input], f)

                  
          N = N + 1    
      i = i + 1               
                                        