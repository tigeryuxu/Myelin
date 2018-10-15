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

import nibabel as nib


"""  Network Begins:
"""
# for input
#input_path = '/project/6015947/yxu233/MyelinUNet_new/create_training/Output/'

input_path = './'

""" Load filenames from zip """
myzip, onlyfiles_mask, counter = read_zip_names(input_path, 'validation_11.zip')

counter = list(range(len(onlyfiles_mask)))  # create a counter, so can randomize it
counter = np.array(counter)
np.random.shuffle(counter)

for i in range(len(onlyfiles_mask)):
    
    filename = onlyfiles_mask[counter[i]]
    input_im, truth_im = load_training_ZIP(myzip, filename)
   
    input_im = nib.Nifti1Image(np.expand_dims(input_im, 0), affine=None)
    truth_im = nib.Nifti1Image(np.expand_dims(truth_im, 0), affine=None)
    

    nib.save(input_im, 'myelin_single_060818_' + "%07d" % (i,) + '_input.nii.gz')
    nib.save(truth_im, 'myelin_single_060818_' + "%07d" % (i,) + '_truth.nii.gz')
    
