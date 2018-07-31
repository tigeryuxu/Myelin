# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:27:54 2018

@author: Tiger
"""

#import tensorflow as tf
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

import pandas as pd
import seaborn as sns

from plot_functions import *
from data_functions import *
from post_process_functions import *
from UNet import *


def calc_jaccard(y, y_b):
    """ Jaccard
    """
    output = np.array(np.argmax(y,axis=-1), dtype=np.float32)
    truth = np.array(np.argmax(y_b,axis=-1), dtype=np.float32)
        
    intersection = np.sum(np.sum(np.multiply(output, truth), axis=-1),axis=-1)
    union = np.sum(np.sum(np.add(output, truth)>= 1, dtype=np.float32),axis=-1) + 0.0000001
    
    jaccard = np.sum(intersection / union)   

    return jaccard


def plot_compare():
    
    categories = ['H1_v_UNet', 'H1_v_H2', 'H1_v_H3', 'H1_v_MATLAB'];
    
    jaccard_path = 'C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Jaccard_testing/all_jaccard/'
    onlyfiles_jaccard = read_file_names(jaccard_path)
    
    #plt.figure()   
    
    newDF = pd.DataFrame()
    for i in range(len(onlyfiles_jaccard)):
        all_jaccard = load_pkl(jaccard_path, onlyfiles_jaccard[i])        
    
        y = [i] * len(all_jaccard)


        df_ = pd.DataFrame(all_jaccard, index=y, columns=categories[i:i+1])
        newDF = newDF.append(df_, ignore_index = True)
        #plt.scatter(all_jaccard, y)

    plt.rcParams.update({'font.size': 18})   
    #save_pkl(all_jaccard, '', 'all_jaccard' + name + '.pkl')
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("whitegrid")
    plt.figure()
    sns.stripplot( data=newDF, jitter=True, orient='h', order=['H1_v_UNet', 'H1_v_H2', 'H1_v_H3', 'H1_v_MATLAB']);
    plt.xlabel('Jaccard')    

    plt.figure()
    sns.violinplot( data=newDF, jitter=True, orient='h', order=['H1_v_UNet', 'H1_v_H2', 'H1_v_H3', 'H1_v_MATLAB']);
    plt.xlabel('Jaccard')    

    plt.figure()
    sns.boxplot( data=newDF, orient='h', order=['H1_v_UNet', 'H1_v_H2', 'H1_v_H3', 'H1_v_MATLAB']);
    plt.xlabel('Jaccard')    
    
    
def find_total_jacc():
    bin_TRUTH = np.copy(TRUTH_fibers)
    bin_TRUTH[bin_TRUTH > 0] = 1
    make_TRUTH = np.zeros([8208, 8208, 2])
    background = np.copy(bin_TRUTH)
    background[background == 1] = 2
    background[background == 0] = 1
    background[background == 2] = 0
    make_TRUTH[:,:,0] = background
    make_TRUTH[:, :, 1] = bin_TRUTH
    
    
    bin_TEST = np.copy(TEST_fibers)
    bin_TRUTH[bin_TEST > 0] = 1
    make_test = np.zeros([8208, 8208, 2])
    background = np.copy(bin_TEST)
    background[background == 1] = 2
    background[background == 0] = 1
    background[background == 2] = 0
    make_test[:,:,0] = background
    make_test[:, :, 1] = bin_TEST
    
    total_jaccard = calc_jaccard(make_TRUTH, make_test)
    
    
def check_jacc_by_fiber(TRUTH_fibers, TEST_fibers, fiber_num_TRUTH):
    
    all_jaccard = []
    for i in range(len(fiber_num_TRUTH)):
        fiber_idx = fiber_num_TRUTH[i]
        
        tmp_TRUTH = np.copy(TRUTH_fibers)
        tmp_TRUTH[tmp_TRUTH != fiber_idx] = 0
        
        tmp_TEST = np.copy(TEST_fibers)
        # MASK the test fibers by tmp_TRUTH, and find all the "cell_num" in the "tmp_TEST"
        tmp_TEST[tmp_TRUTH == 0] = 0
        """ ALSO IF NO FIBERS, THEN SKIP"""
        
        
        numbs = np.unique(tmp_TEST)
        
        tmp_TEST_check = np.zeros(np.shape(tmp_TEST))
        for i in range(len(numbs)):
            cell_idx = numbs[i]
            tmp_TEST_check[tmp_TEST == cell_idx] = 1
            
        """ make test """
        make_test = np.zeros([8208, 8208, 2])
        background = np.copy(tmp_TEST_check)
        background[background == 1] = 2
        background[background == 0] = 1
        background[background == 2] = 0
        make_test[:,:,0] = background
        make_test[:, :, 1] = tmp_TEST_check            
        
        """ make truth """
        make_TRUTH = np.zeros([8208, 8208, 2])
        background = np.copy(tmp_TRUTH)
        background[background == 1] = 2
        background[background == 0] = 1
        background[background == 2] = 0
        make_TRUTH[:,:,0] = background
        make_TRUTH[:, :, 1] = tmp_TRUTH
        
        jaccard = calc_jaccard(make_test, make_TRUTH)
        
        all_jaccard.append(jaccard)
    
    y = [1] * len(all_jaccard)
    
    plt.figure()
    plt.scatter(all_jaccard, y)
    
    save_pkl(all_jaccard, '', 'all_jaccard' + name + '.pkl')        
            

""" Get single Jaccard value for whole image """
def get_global_jacc(back_TRUTH_fibers,back_TEST_fibers):
    TRUTH_fibers = back_TRUTH_fibers
    TRUTH_fibers[TRUTH_fibers > 0] = 1
    make_truth = np.zeros([8208, 8208, 2])
    background = np.copy(TRUTH_fibers)
    background[background == 1] = 2
    background[background == 0] = 1
    background[background == 2] = 0
    make_truth[:,:,0] = background
    make_truth[:, :, 1] = TRUTH_fibers
    
    
    TEST_fibers = back_TEST_fibers
    TEST_fibers[TEST_fibers > 0] = 1
    make_test = np.zeros([8208, 8208, 2])
    background = np.copy(TEST_fibers)
    background[background == 1] = 2
    background[background == 0] = 1
    background[background == 2] = 0
    make_test[:,:,0] = background
    TEST_fibers[TEST_fibers > 0] = 1
    make_test[:, :, 1] = TEST_fibers
    
    global_jacc = calc_jaccard(make_test, make_truth)
    
    return global_jacc


name = 'D_v_AI_301000'
machine = 1
MATLAB = 0


TRUTH_path = 'C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Jaccard_testing/Expert_Truth/'
TESTER_path = 'C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Jaccard_testing/Tester/'
               
""" Read in images to analyze """
onlyfiles_TRUTH = read_file_names(TRUTH_path)
onlyfiles_TEST = read_file_names(TESTER_path)

""" Load images (TRUTH)
"""
#TRUTH_fibers = readIm_counter(TRUTH_path,onlyfiles_TRUTH, 1) 
TRUTH_fibers = load_pkl(TRUTH_path, onlyfiles_TRUTH[1])
back_TRUTH_fibers = np.copy(TRUTH_fibers)

TRUTH_fibers = Image.fromarray(TRUTH_fibers)
fiber_num_TRUTH = np.unique(TRUTH_fibers)

TRUTH_DAPI = readIm_counter(TRUTH_path,onlyfiles_TRUTH, 0)
TRUTH_DAPI_tmp = np.asarray(TRUTH_DAPI, dtype=float)        

#plt.imsave('DAPI_truth.tif', (TRUTH_DAPI_tmp * 255).astype(np.uint16))


""" Load images (TESTER)
"""
#TEST_fibers = readIm_counter(TESTER_path,onlyfiles_TEST, 1) 
TEST_fibers = load_pkl(TESTER_path, onlyfiles_TEST[1])
back_TEST_fibers = np.copy(TEST_fibers)

fiber_num = np.unique(TEST_fibers)
TEST_fibers = Image.fromarray(TEST_fibers)

""" must load pickle DAPI if it's a machine """
if machine:
    """ ALSO don't augment the intensity value by 1 """
    TEST_DAPI_tmp = load_pkl(TESTER_path, onlyfiles_TEST[0])
    
    binary = np.copy(TEST_DAPI_tmp)
    for T in range(len(fiber_num)):
        binary[binary == fiber_num[T]] = -88
    
    binary[binary > 0] = 0
    binary[binary == -88] = 1
    
    
    TEST_DAPI_tmp[binary == 0] = 0
    
#    plt.imsave('TEST_DAPI_OLD' + str(1) + '.tif', (TEST_DAPI_tmp * 255).astype(np.uint16))    
    
    TEST_DAPI = Image.fromarray(TEST_DAPI_tmp)    
    
else:
    TEST_DAPI = readIm_counter(TESTER_path,onlyfiles_TEST, 0)
    TEST_DAPI_tmp = np.asarray(TEST_DAPI, dtype=float)            

    """ maybe try this with final skeletonied and dilated fibers as well??? """    
#plt.imsave('DAPI_test.tif', (TEST_DAPI_tmp * 255).astype(np.uint16))


""" Get single jaccard value for the WHOLE IMAGE"""
global_jacc = get_global_jacc(back_TRUTH_fibers,back_TEST_fibers)



""" Then find DAPI that overlap """
binary_TRUTH = np.copy(TRUTH_DAPI_tmp)
binary_TRUTH[binary_TRUTH > 0] = 1

binary_TEST = np.copy(TEST_DAPI_tmp)
binary_TEST[binary_TEST > 0] = 1

overlap = binary_TRUTH + binary_TEST

overlap[overlap == 1] = 0
overlap[overlap == 2] = 1 

##plt.imsave('overlap.tif', (overlap * 255).astype(np.uint16))
    
""" Then use the overlap to mask the original images """
TRUTH_DAPI_tmp[overlap < 1] = 0
TEST_DAPI_tmp[overlap < 1] = 0

labelled = measure.label(overlap)
cc_TRUTH = measure.regionprops(labelled, intensity_image=TRUTH_DAPI_tmp)    
cc_TEST = measure.regionprops(labelled, intensity_image=TEST_DAPI_tmp)    

all_jaccard = []
for i in range(len(cc_TRUTH)):
    
    DAPI_idx = cc_TRUTH[i]['centroid']
    val = cc_TRUTH[i]['MaxIntensity'] + 1
           
    # extract CROP outo of everything          
    TRUTH_crop, coords = adapt_crop_DAPI(TRUTH_fibers, DAPI_idx, length=1024, width=640)
    TRUTH_crop[TRUTH_crop != val] = 0
    TRUTH_crop[TRUTH_crop == val] = 1
    plt.figure(1); plt.clf(); plt.imshow(TRUTH_crop); plt.pause(0.05)
    
    make_truth = np.zeros([1024, 640, 2])
    background = np.copy(TRUTH_crop)
    background[background == 1] = 2
    background[background == 0] = 1
    background[background == 2] = 0
    make_truth[:,:,0] = background
    make_truth[:, :, 1] = TRUTH_crop
    
    
    DAPI_idx = cc_TEST[i]['centroid']
    val = cc_TEST[i]['MaxIntensity'] + 1
    if machine or MATLAB:
        val = val - 1
    TEST_crop, coords = adapt_crop_DAPI(TEST_fibers, DAPI_idx, length=1024, width=640)
    TEST_crop[TEST_crop != val] = 0
    TEST_crop[TEST_crop == val] = 1
    plt.figure(2); plt.clf(); plt.imshow(TEST_crop); plt.pause(0.05)         

    make_test = np.zeros([1024, 640, 2])
    background = np.copy(TEST_crop)
    background[background == 1] = 2
    background[background == 0] = 1
    background[background == 2] = 0
    make_test[:,:,0] = background
    make_test[:, :, 1] = TEST_crop
    
    jaccard = calc_jaccard(make_test, make_truth)
    
    all_jaccard.append(jaccard)

y = [1] * len(all_jaccard)

plt.figure()
plt.scatter(all_jaccard, y)

save_pkl(all_jaccard, '', 'all_jaccard' + name + '.pkl')








    
    

