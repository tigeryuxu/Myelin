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
import bz2

from plot_functions import *
from data_functions import *
from pre_processing import *

# for saving
s_path = 'D:/Tiger/AI stuff/OUTPUT_MATLAB_2/'

## for training
#input_path='D:/Tiger/AI stuff/Input/'
#DAPI_path='D:/Tiger/AI stuff/DAPI_masks/'
#mask_path='D:/Tiger/AI stuff/Fibers_masks/'

import tkinter
from tkinter import filedialog
root = tkinter.Tk()
input_path = filedialog.askdirectory(parent=root, initialdir="/",
                                    title='Please select input directory')
input_path = input_path + '/'

mask_path = filedialog.askdirectory(parent=root, initialdir=input_path,
                                    title='Please select masks directory')
mask_path = mask_path + '/'
#s_path = filedialog.askdirectory(parent=root, initialdir=input_path,
#                                    title='Please select saving directory')

# Read in file names
onlyfiles_mask = [ f for f in listdir(mask_path) if isfile(join(mask_path,f))]   

# ONLY FOR August_18
#newfiles_mask = []
#for filename in onlyfiles_mask:
#    split_n = filename.split('3', 1)
#    print(split_n[0] + split_n[1])
#    newfiles_mask.append(split_n[0] + '_' + split_n[1])
#onlyfiles_mask = newfiles_mask

natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
onlyfiles_mask.sort(key = natsort_key1)

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
j = 0
k = 0
while i < len(counter): #length of your filename      
      input_arr = readIm_counter(input_path,onlyfiles_test, counter[k])
      DAPI_arr = readIm_counter(input_path,onlyfiles_test, counter[k + 3])
      DAPI_arr = np.asarray(DAPI_arr)
      DAPI = DAPI_arr[0:1440, 0:1920, 2]
      #labels = DAPI_count(DAPI)  # count DAPI + do watershed
      k = k + 5

      DAPI_arr = readIm_counter(mask_path,onlyfiles_mask, counter[j])
      j = j + 1;
      mask_arr = readIm_counter(mask_path,onlyfiles_mask, counter[j])
      j = j + 1;
    
      DAPI_tmp = np.asarray(DAPI_arr, dtype=float)
      labelled = measure.label(DAPI_tmp)
      cc = measure.regionprops(labelled)
      
      # SHOULD RANDOMIZE THE COUNTER      
      counter_DAPI = list(range(len(cc)))  # create a counter, so can randomize it
      counter_DAPI = np.array(counter_DAPI)
      #np.random.shuffle(counter_DAPI)
      N = 0
      while N < len(cc):  
          print(N)
          DAPI_idx = cc[counter_DAPI[N]]['centroid']
          
          length = 1440
          width = 1920
          # extract CROP outo of everything          
          DAPI_crop, coords = adapt_crop_DAPI(DAPI_arr, DAPI_idx, length=length, width=width)                    
          truth_crop, coords = adapt_crop_DAPI(mask_arr, DAPI_idx, length=length, width=width)
          input_crop, coords = adapt_crop_DAPI(input_arr, DAPI_idx, length=length, width=width)         
          
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
          
          concate_input = np.zeros([length, width, 5])
          
          concate_input[:, :, 0:3] = input_crop
          #concate_input[:, :, 0] = zeros(np.shape(concate_input))
          concate_input[:, :, 3:5] = combined
          
          #DAPI = np.copy(concate_input[:, :, 2])
          #DAPI[DAPI > 0] = 1
          show = np.copy(concate_input[:, :, 0:3])
          #show[:, :, 2] = labels
          clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
          cl1 = clahe.apply(DAPI)
          show[:, :, 2] = cl1
          
          plt.figure('Batched_check'); plt.clf()
          plt.subplot(221); plt.imshow(show/255);
          true_m = np.argmax((concate_input[:, :, 3:5]).astype('uint8'), axis=-1)  
          plt.subplot(222); plt.imshow(true_m, vmin = 0, vmax = 1);      
          
          plt.subplot(223); plt.imshow(DAPI); plt.pause(0.00001)


          """ Saves image """        
          # Saving the objects:
          filename = onlyfiles_mask[counter[j - 1]]
          split_n = filename.split('.')
#          with open(s_path + split_n[0] + '_' + str(N) + '_RERUN', 'wb') as f:  # Python 3: open(..., 'wb')
#               pickle.dump([concate_input], f)
          #print(filename)
          with bz2.BZ2File(s_path + split_n[0] + '_' + str(N), 'wb') as f:
            pickle.dump([concate_input], f)

#          import gzip
#         import cPickle as pickle
#          with gzip.GzipFile(s_path + split_n[0] + '_' + str(N), 'w') as f:
#            pickle.dump([concate_input], f
          
          if N % 8 == 0:
              pass
              plt.savefig(s_path + split_n[0] + str(N) + split_n[1], dpi = 200)
              plt.close()
                   
          N = N + 1    
      i = i + 1               
                                        