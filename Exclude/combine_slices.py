"""
Created on Tue Jan  2 12:29:40 2018

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

from skimage import data, exposure, img_as_float

import tkinter
from tkinter import filedialog
""" Network Begins:
"""
scale = 0.519  #0.519, 0.6904, 0.35
minLength = 15 / scale
minSingle = 50 / scale
minLengthDuring = 4/scale
radius = 3/scale  # um

len_x = 1024     # 1344, 1024
width_x = 640   # 864, 640



root = tkinter.Tk()
input_path = filedialog.askdirectory(parent=root, initialdir="/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Source/",
                                    title='Please select input directory')
input_path = input_path + '/'

""" Pre-processing """
# Read in file names
onlyfiles_mask = [ f for f in listdir(input_path) if isfile(join(input_path,f))]   
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
onlyfiles_mask.sort(key = natsort_key1)

counter = list(range(len(onlyfiles_mask)))  # create a counter, so can randomize it

# Read in file names
onlyfiles_mask = [ f for f in listdir(input_path) if isfile(join(input_path,f))]   
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
onlyfiles_mask.sort(key = natsort_key1)

counter = list(range(len(onlyfiles_mask)))  # create a counter, so can randomize it

batch_x = []; batch_y = [];

row = 0
col = 0
count_eight = 0
new_im = []
for i in range(len(onlyfiles_mask)):  
    total_counter = 0
    filename = onlyfiles_mask[counter[i]]
     

    """ Load image """
    input_arr = readIm_counter(input_path,onlyfiles_mask, counter[i]) 
    size_whole = input_arr.size[0]
    
    """ save the stitched image """
    if i % 16 == 0 and i != 0:
        name = onlyfiles_mask[counter[i - 5]]
        new_im = np.asarray(new_im, dtype='uint8')
        plt.imsave('COMB_SLICE_' + str(i) + '_' + name, new_im)
    
    if i % 16 == 0:    
        new_im = np.zeros([size_whole * 4, size_whole * 4, 3])
        row = 0
        col = 0
        count_eight = 0
        
    new_im[row * size_whole: (row + 1) * size_whole, col * size_whole: (col + 1) * size_whole] = input_arr
        
    print(row, col)

    count_eight = i % 8
    
    if count_eight < 3:
        col = col + 1
    
    elif count_eight == 3:
        col = col
        
    elif count_eight > 3 and count_eight != 7:
        col = col - 1
        
        
    if count_eight == 3 or count_eight == 7:
        row = row + 1
    
    
    
    
    
    
    
    