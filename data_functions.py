# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:25:15 2017

@author: Tiger
"""

""" Retrieves validation images
"""

import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from skimage import measure
from natsort import natsort_keygen, ns
import os
import pickle
import scipy.io as sio

import zipfile
import bz2

from plot_functions import *
from data_functions import *
from post_process_functions import *
from UNet import *



def csv_num_sheaths_violin():
    
    categories = ['H1_v_AI', 'H1_v_H2', 'H1_v_H3', 'H1_v_MATLAB'];

    categories = ['H1_v_AI', 'H1_v_H2', 'e'];
      
    import csv
    with open('num_sheaths_data.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        newDF = pd.DataFrame()  

        i = 0        
        for row in spamreader:         

            row_int = []
            print (', '.join(row))
            for t in row[0]:
                if t != ',':
                    row_int.append(int(t))

            y = [i] * len(row_int)
                  
            df_ = pd.DataFrame(row_int, index=y, columns=categories[i+1:i+2])
            newDF = newDF.append(df_, ignore_index = True)

            i = i + 1
    #save_pkl(all_jaccard, '', 'all_jaccard' + name + '.pkl')
    plt.figure()
    sns.stripplot( data=newDF, jitter=True, orient='h');
    plt.xlabel('Jaccard')    

    plt.figure()
    sns.violinplot( data=newDF, jitter=True, orient='h');
    plt.xlabel('Jaccard')    

    plt.figure()
    sns.boxplot( data=newDF, orient='h');
    plt.xlabel('Jaccard')  

""" Parses through the validation zip to return counters that index to which files have
    fibers and which do not
    (speeds up processing time for batch so don't have to do this every time)    
"""
def parse_validation(myzip_val, onlyfiles_val, counter_val):
    
    counter_fibers = []
    counter_blank = []
    for T in range(len(counter_val)):
        """ Get validation images """
        index = counter_val[T]
        filename = onlyfiles_val[index]
        
        input_val, truth_val = load_training_ZIP(myzip_val, filename)
        
        """ Check to see if contains fibers or not """
        if np.count_nonzero(truth_val[:, :, 1]) > 0:     # IF YES, there are fibers
            counter_fibers.append(T)

        elif np.count_nonzero(truth_val[:, :, 1]) == 0:
            counter_blank.append(T)
            
            
    return counter_fibers, counter_blank


""" Parses through the validation zip to return counters that index to which files have
    fibers and which do not
    (speeds up processing time for batch so don't have to do this every time)    
"""
def parse_validation_QL(myzip_val, onlyfiles_val, counter_val):
    
    counter_fibers = []
    counter_blank = []
    for T in range(len(counter_val)):
        """ Get validation images """
        index = counter_val[T]
        filename = onlyfiles_val[index]
        
        input_val, truth_val = load_training_ZIP_Zpickle(myzip_val, filename)
        
        """ Check to see if contains fibers or not """
        if np.count_nonzero(truth_val[:, :, 1]) > 0:     # IF YES, there are fibers
            counter_fibers.append(T)

        elif np.count_nonzero(truth_val[:, :, 1]) == 0:
            counter_blank.append(T)
            
            
    return counter_fibers, counter_blank



""" changes QL images to cropped size """
def check_shape_QL(input_im, truth_im, len_im, width_im):
    input_arr = Image.fromarray(input_im.astype(np.uint8))
    
    resized = resize_adaptive(input_arr, 1500, method=Image.BICUBIC)
    resized_arr = np.asarray(resized)
    
    labelled = measure.label(resized_arr[:,:, 1])
    cc = measure.regionprops(labelled)
                   
    DAPI_idx = cc[0]['centroid']            
    # extract CROP outo of everything     
    len_im = 1024
    width_im = 640
    input_crop, coords = adapt_crop_DAPI(resized, DAPI_idx, length=len_im, width=width_im)
    
    """ resize the truth as well """
    truth_resized = Image.fromarray(truth_im[:, :, 1])
    truth_resized = resize_adaptive(truth_resized, 1500, method=Image.BICUBIC)
    
    truth_crop, coords_null = adapt_crop_DAPI(truth_resized, DAPI_idx, length=len_im, width=width_im)
    truth_crop[truth_crop > 0] = 1
    
    truth_whole = np.ones([len_im, width_im, 2])
    truth_null = truth_whole[:,:,0]
    
    truth_null[truth_crop == 1] = 0
    truth_whole[:,:,1] = truth_crop
    truth_whole[:,:,0] = truth_null
    
    truth_im = truth_whole
    input_im = input_crop
    
    return input_im, truth_im

""" returns a batch of validation images from the zip file """

def get_batch_val(myzip, onlyfiles_val, counter,  mean_arr, std_arr, batch_size=2):
    
    np.random.shuffle(counter)
    batch_x_val = [];
    batch_y_val = [];
    batch_weights = [];
    for T in range(int(batch_size)):
        """ Get validation images """
        index = counter[T]
        filename = onlyfiles_val[index]
        
        input_val, truth_val = load_training_ZIP(myzip, filename)
        
        if input_val.shape[1] > 1500:
            input_val, truth_val = check_shape_QL(input_val, truth_val, len_im=1024, width_im=640)
        
        
        """ Normalize the image first """
        input_val = normalize_im(input_val, mean_arr, std_arr) 
        
        
        fiber_label = np.copy(truth_val[:, :, 1])
        """ Make spatial weighting map """
        sp_weighted_labels = spatial_weight(fiber_label,edgeFalloff=10,background=0.01,approximate=True)

        """ Create a matrix of weighted labels """
        weighted_labels = np.copy(truth_val)
        weighted_labels[:, :, 1] = sp_weighted_labels
        
        
        
        batch_x_val.append(input_val)
        batch_y_val.append(truth_val)
        batch_weights.append(weighted_labels)
                    
    return batch_x_val, batch_y_val, batch_weights

""" returns a batch of validation images from the zip file """
def get_batch_val_QL(myzip, onlyfiles_val, counter,  mean_arr, std_arr, batch_size=2):
    
    np.random.shuffle(counter)
    batch_x_val = [];
    batch_y_val = [];
    batch_weights = [];
    for T in range(int(batch_size)):
        """ Get validation images """
        index = counter[T]
        filename = onlyfiles_val[index]
        
        input_val, truth_val = load_training_ZIP_Zpickle(myzip, filename)
    
        input_val = input_val[0:1440, 0:1920, :]
        truth_val = truth_val[0:1440, 0:1920, :]    
        """ Normalize the image first """
        input_val = normalize_im(input_val, mean_arr, std_arr) 
        
        
        fiber_label = np.copy(truth_val[:, :, 1])
        """ Make spatial weighting map """
        sp_weighted_labels = spatial_weight(fiber_label,edgeFalloff=10,background=0.01,approximate=True)

        """ Create a matrix of weighted labels """
        weighted_labels = np.copy(truth_val)
        weighted_labels[:, :, 1] = sp_weighted_labels
        
        
        
        batch_x_val.append(input_val)
        batch_y_val.append(truth_val)
        batch_weights.append(weighted_labels)
                    
    return batch_x_val, batch_y_val, batch_weights


""" returns a batch of validation images from the zip file """
def get_batch_val_bz(input_path, onlyfiles_val, counter,  mean_arr, std_arr, batch_size=2):
    
    np.random.shuffle(counter)
    batch_x_val = [];
    batch_y_val = [];
    batch_weights = [];
    for T in range(int(batch_size)):
        """ Get validation images """
        index = counter[T]
        filename = onlyfiles_val[index]
        
        input_val, truth_val = load_training_bz(input_path, filename)
    
        input_val = input_val[0:1440, 0:1920, :]
        truth_val = truth_val[0:1440, 0:1920, :]    
        """ Normalize the image first """
        input_val = normalize_im(input_val, mean_arr, std_arr) 
        
        
        fiber_label = np.copy(truth_val[:, :, 1])
        """ Make spatial weighting map """
        sp_weighted_labels = spatial_weight(fiber_label,edgeFalloff=10,background=0.01,approximate=True)

        """ Create a matrix of weighted labels """
        weighted_labels = np.copy(truth_val)
        weighted_labels[:, :, 1] = sp_weighted_labels
        
        
        
        batch_x_val.append(input_val)
        batch_y_val.append(truth_val)
        batch_weights.append(weighted_labels)
                    
    return batch_x_val, batch_y_val, batch_weights


""" returns a batch of validation images from the zip file """
def get_batch_val_normal(input_path, onlyfiles_val, counter,  mean_arr, std_arr, batch_size=2):
    
    np.random.shuffle(counter)
    batch_x_val = [];
    batch_y_val = [];
    batch_weights = [];
    for T in range(int(batch_size)):
        """ Get validation images """
        index = counter[T]
        filename = onlyfiles_val[index]
        
        input_val, truth_val = load_training(input_path, filename)
    
        input_val = input_val[0:1440, 0:1920, :]
        truth_val = truth_val[0:1440, 0:1920, :]    
        """ Normalize the image first """
        input_val = normalize_im(input_val, mean_arr, std_arr) 
        
        
        fiber_label = np.copy(truth_val[:, :, 1])
        """ Make spatial weighting map """
        sp_weighted_labels = spatial_weight(fiber_label,edgeFalloff=10,background=0.01,approximate=True)

        """ Create a matrix of weighted labels """
        weighted_labels = np.copy(truth_val)
        weighted_labels[:, :, 1] = sp_weighted_labels
        
        
        
        batch_x_val.append(input_val)
        batch_y_val.append(truth_val)
        batch_weights.append(weighted_labels)
                    
    return batch_x_val, batch_y_val, batch_weights

""" Returns names of all files in zipfile 
    Also returns counter to help with randomization
"""
def read_file_names(input_path):    
    # Read in file names
    onlyfiles = [ f for f in listdir(input_path) if isfile(join(input_path,f))]   
    natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
    onlyfiles.sort(key = natsort_key1)

    return onlyfiles


""" Returns names of all files in zipfile 
    Also returns counter to help with randomization
"""
def read_zip_names(input_path, filename):    
    # Read in file names
    myzip = zipfile.ZipFile(input_path + filename, 'r')
    onlyfiles = myzip.namelist()
    natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
    onlyfiles.sort(key = natsort_key1)
    counter = list(range(len(onlyfiles)))  # create a counter, so can randomize it

    return myzip, onlyfiles, counter


""" Saving the objects """
def save_pkl(obj_save, s_path, name):
    with open(s_path + name, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([obj_save], f)

"""Getting back the objects"""
def load_pkl(s_path, name):
    with open(s_path + name, 'rb') as f:  # Python 3: open(..., 'rb')
      loaded = pickle.load(f)
      obj_loaded = loaded[0]
      return obj_loaded


""" Load training data from zip archive """
def load_training_ZIP(myzip, filename):
    contents = pickle.load(myzip.open(filename))
    concate_input = contents[0]
    
    if np.shape(concate_input)[-1] < 6:   # originally only had 5
        input_im =concate_input[:, :, 0:3]
        truth_im =concate_input[:, :, 3:5]    
    else:                                 # now have extra fiber channel
        input_im =concate_input[:, :, 0:4]
        truth_im =concate_input[:, :, 4:6]    

    return input_im, truth_im
        

""" Load training data from zip archive """
def load_training_ZIP_Zpickle(myzip, filename):
    tmp = myzip.open(filename)
    contents = []
    with bz2.open(tmp, 'rb') as f:
        loaded_object = pickle.load(f)
        contents = loaded_object[0]
    concate_input = contents
    
    if np.shape(concate_input)[-1] < 6:   # originally only had 5
        input_im =concate_input[:, :, 0:3]
        truth_im =concate_input[:, :, 3:5]    
    else:                                 # now have extra fiber channel
        input_im =concate_input[:, :, 0:4]
        truth_im =concate_input[:, :, 4:6]    

    return input_im, truth_im

""" Load training data """
def load_training(s_path, filename):

    # Getting back the objects:
    with open(s_path + filename, 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        concate_input = loaded[0]

    if np.shape(concate_input)[-1] < 6:   # originally only had 5
        input_im =concate_input[:, :, 0:3]
        truth_im =concate_input[:, :, 3:5]    
    else:                                 # now have extra fiber channel
        input_im =concate_input[:, :, 0:4]
        truth_im =concate_input[:, :, 4:6]    
    
    return input_im, truth_im

""" Load training zipped data """
def load_training_bz(s_path, filename):

    # Getting back the objects:
    contents = []
    with bz2.open(s_path + filename, 'rb') as f:
        loaded_object = pickle.load(f)
        contents = loaded_object[0]
    concate_input = contents

    if np.shape(concate_input)[-1] < 6:   # originally only had 5
        input_im =concate_input[:, :, 0:3]
        truth_im =concate_input[:, :, 3:5]    
    else:                                 # now have extra fiber channel
        input_im =concate_input[:, :, 0:4]
        truth_im =concate_input[:, :, 4:6]    
    
    return input_im, truth_im
        
        

def get_validate(test_input_path, DAPI_path, mask_path, mean_arr, std_arr):
        
  batch_x = []
  batch_y = []
  # Read in file names
  onlyfiles_mask = [ f for f in listdir(mask_path) if isfile(join(mask_path,f))]   
  natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
  onlyfiles_mask.sort(key = natsort_key1)

  # Read in file names
  onlyfiles_DAPI = [ f for f in listdir(DAPI_path) if isfile(join(DAPI_path,f))]
  onlyfiles_DAPI.sort(key = natsort_key1)
      
  # Read in truth image names
  onlyfiles_test = [ f for f in listdir(test_input_path) if isfile(join(test_input_path,f))] 
  onlyfiles_test.sort(key = natsort_key1)    
    
  input_arr = readIm_counter(test_input_path,onlyfiles_test, 0)
  DAPI_arr = readIm_counter(DAPI_path,onlyfiles_DAPI, 0)
  mask_arr = readIm_counter(mask_path,onlyfiles_mask, 0)
            
  """
      Then split the DAPI into pixel idx list
      then cycle through each DAPI
      then CROP a rectangle around the input_arr, the DAPI_mask, AND the fibers_mask
      
      find corresponding fibers by adding (+1) to the value of the DAPI pixel
      then concatenate DAPI_mask to the input_arr
  """
  #DAPI_arr[DAPI_arr > 0] = 1
  DAPI_tmp = np.asarray(DAPI_arr, dtype=float)
  labelled = measure.label(DAPI_tmp)
  cc = measure.regionprops(labelled)
  
  # SHOULD RANDOMIZE THE COUNTER      
  counter_DAPI = list(range(len(cc)))  # create a counter, so can randomize it
  counter_DAPI = np.array(counter_DAPI)
  np.random.shuffle(counter_DAPI)
  
  N = 0
  while N < len(counter_DAPI):  
      DAPI_idx = cc[counter_DAPI[N]]['centroid']
      
      # extract CROP outo of everything          
      DAPI_crop = adapt_crop_DAPI(DAPI_arr, DAPI_idx, length=704, width=480)                    
      truth_crop = adapt_crop_DAPI(mask_arr, DAPI_idx, length=704, width=480)
      input_crop = adapt_crop_DAPI(input_arr, DAPI_idx, length=704, width=480)          
      
      """ Find fibers (truth_mask should already NOT contain DAPI, so don't need to get rid of it)
          ***however, the DAPI pixel value of DAPI_center should be the SAME as fibers pixel value + 1
      """
      val_at_center = DAPI_tmp[DAPI_idx[0].astype(int), DAPI_idx[1].astype(int)] 
      val_fibers = val_at_center + 1
      
      # Find all the ones that are == val_fibers
      truth_crop[truth_crop != val_fibers] = 0
      truth_crop[truth_crop == val_fibers] = 255
      
      # then split into binary classifier truth:
      fibers = np.copy(truth_crop)
      fibers = np.expand_dims(fibers, axis=3)
      
      null_space = np.copy(truth_crop)
      null_space[null_space == 0] = -1
      null_space[null_space > -1] = 0
      null_space[null_space == -1] = 1
      null_space = np.expand_dims(null_space, axis=3)
      
      combined = np.append(null_space, fibers, -1)
      
      """ Eliminate all other DAPI """
      DAPI_crop[DAPI_crop != val_at_center] = 0
      DAPI_crop[DAPI_crop == val_at_center] = 1
       
      """ Delete green channel by making it the DAPI_mask instead """
      input_crop[:, :, 1] = DAPI_crop

      """ Normalize here"""
      input_crop = normalize_im(input_crop, mean_arr, std_arr)  
      concate_input = input_crop
      
      """ set inputs and truth """
      batch_x.append(concate_input)
      batch_y.append(combined)
              
      N = N + 1
  return batch_x, batch_y






      
""" Adaptive cropping
        Inputs:
            - im ==> original full-scale image
            - DAPI_center ==> idx of center of DAPI point
            - length ==> width of SQUARE around the DAPI point to crop
        if nears an edge (index out of bounds) ==> will crop the opposite direction however many pixels are left  
"""
def adapt_crop_DAPI(im, DAPI_center, length=704, width=480):
    
    # first find limits of image
    w_lim, h_lim = im.size
    
    # then find limits of crop
    top = DAPI_center[0] - length/2
    bottom = DAPI_center[0] + length/2
    left = DAPI_center[1] - width/2
    right = DAPI_center[1] + width/2
    
    """ check if it's even possible to create this square
    """
    total = (bottom - top) * (right - left)
    if total > w_lim * h_lim:
        print("CROP TOO BIG")
        #throw exception
    
    """ if it's possible to create the square, adjust the excess as needed
    """
    if h_lim - bottom < 0:  # out of bounds height
        excess = bottom - h_lim
        bottom = h_lim  # reset the bottom to the MAX
        top = top - excess  # raise the top
    
    if top < 0: # out of bounds height
        excess = top * (-1)
        top = 0
        bottom = bottom + excess
        
    if w_lim - right < 0: # out of bounds width
        excess = right - w_lim
        right = w_lim
        left = left - excess
    
    if left < 0: # out of bounds width
        excess = left * (-1)
        left = 0
        right = right + excess
    
    """ CROP and convert to float array
    """
    cropped_im = im.crop((left, top, right, bottom))
    cropped_im = np.asarray(cropped_im, dtype=float)
    
    coords = [top, bottom, left, right]
    return cropped_im, coords



""" Adaptive cropping FOR ARRAYS
        Inputs:
            - im ==> original full-scale image
            - DAPI_center ==> idx of center of DAPI point
            - length ==> width of SQUARE around the DAPI point to crop
        if nears an edge (index out of bounds) ==> will crop the opposite direction however many pixels are left  
"""
def adapt_crop_DAPI_ARRAY(im, DAPI_center, length=704, width=480):
    
    # first find limits of image
    w_lim, h_lim = im.shape
    
    # then find limits of crop
    top = DAPI_center[0] - length/2
    bottom = DAPI_center[0] + length/2
    left = DAPI_center[1] - width/2
    right = DAPI_center[1] + width/2
    
    """ check if it's even possible to create this square
    """
    total = (bottom - top) * (right - left)
    if total > w_lim * h_lim:
        print("CROP TOO BIG")
        #throw exception
    
    """ if it's possible to create the square, adjust the excess as needed
    """
    if h_lim - bottom < 0:  # out of bounds height
        excess = bottom - h_lim
        bottom = h_lim  # reset the bottom to the MAX
        top = top - excess  # raise the top
    
    if top < 0: # out of bounds height
        excess = top * (-1)
        top = 0
        bottom = bottom + excess
        
    if w_lim - right < 0: # out of bounds width
        excess = right - w_lim
        right = w_lim
        left = left - excess
    
    if left < 0: # out of bounds width
        excess = left * (-1)
        left = 0
        right = right + excess
    

    """ Ensure within boundaries """
    top = int(top)
    bottom = int(bottom)
    right = int(right)
    left = int(left)
    add_l = length - (bottom - top)
    add_w = width - (right - left)
    
    if add_l: bottom = bottom + add_l
    if add_w: right = right + add_w
    
    """ CROP """    
    cropped_im = im[int(left):int(right), int(top):int(bottom)] 
    #cropped_im = np.asarray(cropped_im, dtype=float)
    
    coords = [top, bottom, left, right]
    return cropped_im, coords

"""
    Find standard deviation + mean
"""
def calc_avg_mod(input_path, onlyfiles_mask):


    array_of_ims = []    
    for i in range(len(onlyfiles_mask)):
        filename = onlyfiles_mask[i]
        input_im, truth_im = load_training(input_path, filename)
    
        array_of_ims.append(input_im)
        
    mean_arr = np.mean(array_of_ims)  # saves the mean_arr to be used for cross-validation
    array_of_ims = array_of_ims - mean_arr
    std_arr = np.std(array_of_ims)  # saves the std_arr to be used for cross-validation
    array_of_ims = array_of_ims / std_arr 


    return mean_arr, std_arr      



"""
    Find standard deviation + mean
"""
def calc_avg(input_path):
    
    cwd = os.getcwd()     # get current directory
    os.chdir(input_path)   # change path
    
    # Access all PNG files in directory
    allfiles=os.listdir(os.getcwd())
    imlist=[filename for filename in allfiles if  filename[-4:] in [".tif",".TIF"]]

    array_of_ims = []
    # Build up list of images that have been casted to float
    for im in imlist:
        im_org = Image.open(im)
        im_res = resize(im_org, h=4104, w=4104)
        imarr=np.array(im_res,dtype=np.float)
        array_of_ims.append(imarr)

    mean_arr = np.mean(array_of_ims)  # saves the mean_arr to be used for cross-validation
    array_of_ims = array_of_ims - mean_arr
    std_arr = np.std(array_of_ims)  # saves the std_arr to be used for cross-validation
    array_of_ims = array_of_ims / std_arr 

    os.chdir(cwd)

    return mean_arr, std_arr      


"""
    To normalize by the mean and std
"""
def normalize_im(im, mean_arr, std_arr):
    normalized = (im - mean_arr)/std_arr 
    return normalized        
    
"""
    if RGB ==> num_dims == 3
"""
def crop_im(mask, num_dims, width, height):
    left = 0; up =  0
    right = width; down = height 
          
    new_c = mask.crop((left, up, right, down))         
    new_c = np.array(new_c)
    if num_dims == 2:
        new_c = np.expand_dims(new_c, axis=3)
    all_crop = new_c
    
    return all_crop

def resize(im, size_h=8208, size_w=8208, method=Image.BICUBIC):
    
    im = im.resize([size_h,size_w], resample=method)
    return im


def resize_adaptive(img, basewidth, method=Image.BICUBIC):
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), resample=method)
    return img


""" Reads in image at path "mypath" and for certain file number "fileNum"
"""
def readIm_counter(mypath, onlyfiles, fileNum, size_h=8208, size_w=8208): 
    curpath = mypath + onlyfiles[fileNum]
    im = Image.open(curpath)      
    return im

""" Reads in image at path "mypath" and for certain file number "fileNum"
"""
def readIm_counter_DAPI(mypath, onlyfiles, fileNum): 
    curpath = mypath + onlyfiles[fileNum]
    im = Image.open(curpath).convert('L')
    return im

""" Reads in image at path "mypath" and for certain file number "fileNum"
"""
def readIm_counter_MATLAB(mypath, onlyfiles, fileNum): 
    curpath = mypath + onlyfiles[fileNum]
    mat_contents = sio.loadmat(curpath)
    mask = mat_contents['save_im']    
    #mask = Image.fromarray(mask)    
    return mask


""" Reads in image at path "mypath" and for certain file number "fileNum"
"""
def readIm_ZIP(mypath, onlyfiles, fileNum,  size_h=8208, size_w=8208): 
    curpath = mypath + onlyfiles[fileNum]
    im = Image.open(curpath)
    im = resize(im, h=size_h, w=size_w)        
    return im