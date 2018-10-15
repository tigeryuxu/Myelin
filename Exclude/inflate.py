# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:32:06 2018

@author: Neuroimmunology Unit
"""

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
import bz2


"""  Network Begins:
"""
#filename  = 'August_2_R'
filename  = 'August_08_B'  # not ready

#filename  = 'August_11_B'
#filename  = 'August_15_B'
#filename  = 'August_16_B'
#filename  = 'August_18_2-3'
#filename  = 'August_21_R'
#filename  = 'August_22_R'
filename  = 'August_25_2-3'


# for input
input_path = 'D:/Tiger/AI stuff/OUTPUT_MATLAB_1/'

#new_path = 'D:/Tiger/AI stuff/OUTPUT_MATLAB_2/'
new_path = 'F:/'

""" Load filenames from zip """
onlyfiles = [ f for f in listdir(input_path) if isfile(join(input_path,f))]   
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
onlyfiles.sort(key = natsort_key1)

onlyfiles_new = []
for blah in onlyfiles:
    if filename not in blah: 
        continue
    onlyfiles_new.append(blah)
onlyfiles = onlyfiles_new

i = 0
while i < len(onlyfiles):
    #input_im, truth_im = load_training_bz(input_path, onlyfiles_mask[i])
    filename = onlyfiles[i]
    contents = []
    with bz2.open(input_path + filename, 'rb') as f:
        loaded_object = pickle.load(f)
        contents = loaded_object[0]
    concate_input = contents
#    if np.shape(concate_input)[-1] < 6:   # originally only had 5
#        input_im =concate_input[:, :, 0:3]
#        truth_im =concate_input[:, :, 3:5]    
#    else:                                 # now have extra fiber channel
#        input_im =concate_input[:, :, 0:4]
#        truth_im =concate_input[:, :, 4:6]   
#    
#    plt.figure(1); plt.clf(); show_norm(input_im); plt.pause(0.00001)
    with open(new_path + filename + '_RERUN', 'wb') as f:  # Python 3: open(..., 'wb')
       pickle.dump([concate_input], f)
       
    print(i)
    
    i = i + 1
    
    
    



