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

import tkinter
from tkinter import filedialog

""" Network Begins:
"""
scale = 0.227
min_microns = 12
minLength = min_microns / scale
minSingle = min_microns / scale
minLengthDuring = 4/scale
radius = 3/scale  # um

len_im = 1024    # 1024, 1440
width_im = 800   # 800, 1920

green = 2  # or 0

root = tkinter.Tk()
input_path = filedialog.askdirectory(parent=root, initialdir="/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Source/",
                                    title='Please select input directory')
input_path = input_path + '/'

#root = tkinter.Tk()
s_path = filedialog.askdirectory(parent=root, initialdir="/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Checkpoints/",
                                    title='Please select checkpoint directory')
s_path = s_path + '/'


sav_dir = filedialog.askdirectory(parent=root, initialdir="/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Source/",
                                    title='Please select saving directory')
sav_dir = sav_dir + '/'



""" Pre-processing """
# Read in file names
onlyfiles_mask = [ f for f in listdir(input_path) if isfile(join(input_path,f))]   
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
onlyfiles_mask.sort(key = natsort_key1)

# Read in file names
#onlyfiles_DAPI = [ f for f in listdir(DAPI_path) if isfile(join(DAPI_path,f))]
#onlyfiles_DAPI.sort(key = natsort_key1)

counter = list(range(len(onlyfiles_mask)))  # create a counter, so can randomize it

""" Load avg_img and std_img from TRAINING SET """
mean_arr = 0; std_arr = 0;
with open('mean_arr.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    loaded = pickle.load(f)
    mean_arr = loaded[0]
         
with open('std_arr.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    loaded = pickle.load(f)
    std_arr = loaded[0]


# Variable Declaration

x = tf.placeholder('float32', shape=[None, len_im, width_im, 3], name='InputImage')
y_ = tf.placeholder('float32', shape=[None, len_im, width_im, 2], name='CorrectLabel')
training = tf.placeholder(tf.bool, name='training')

# Read in file names
#onlyfiles_mask = [ f for f in listdir(input_path) if isfile(join(input_path,f))]   
#natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
#onlyfiles_mask.sort(key = natsort_key1)

# Read in file names
#onlyfiles_DAPI = [ f for f in listdir(DAPI_path) if isfile(join(DAPI_path,f))]
#onlyfiles_DAPI.sort(key = natsort_key1)

#counter = list(range(len(onlyfiles_mask)))  # create a counter, so can randomize it

""" Creates network and cost function"""
y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed = create_network(x, y_, training)
#accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm_CLASSW(y, y_b, logits)

sess = tf.InteractiveSession()
""" TO LOAD OLD CHECKPOINT """
saver = tf.train.Saver()
check_path = ''
saver.restore(sess, s_path + 'check_655000')

batch_x = []; batch_y = [];
k = 0;
for i in range(len(onlyfiles_mask)): 
    
    if k >= len(onlyfiles_mask):
        break;
        
    total_counter = 0
    filename = onlyfiles_mask[counter[k]]
       
    """ Load image
    """
    input_arr = readIm_counter(input_path,onlyfiles_mask, counter[k])    
    if np.shape(input_arr)[1] > 1500:
        input_arr = resize_adaptive(input_arr, 1500, method=Image.BICUBIC)
    
    input_save = np.copy(np.asarray(input_arr))
    if green == 2:
        green_arr = np.copy(input_save[:,:,1])
        input_save[:,:,0] = green_arr
        input_arr = Image.fromarray(input_save)
    DAPI_arr = readIm_counter(input_path,onlyfiles_mask, counter[k + 3])
    if np.shape(DAPI_arr)[1] > 1500:
        DAPI_arr = resize_adaptive(DAPI_arr, 1500, method=Image.BICUBIC)

    red_arr = readIm_counter(input_path,onlyfiles_mask, counter[k + 2 + green])
    if np.shape(red_arr)[1] > 1500:
        red_arr = resize_adaptive(red_arr, 1500, method=Image.BICUBIC)

    k = k + 5
    DAPI_arr = np.asarray(DAPI_arr)
    DAPI = DAPI_arr[:, :, 2]

    red_arr = np.asarray(red_arr)
    red = red_arr[:, :, 0]
    
    if green == 2:
        red = red_arr[:,:,1]
    
#    input_arr = input_arr[0:1440, 0:1920, 2]
    input_arr_tmp = np.zeros(np.shape(input_arr))
    input_arr_tmp[:,:,0] = red
    input_arr_tmp[:,:,2] = DAPI
    #DAPI_tmp = np.asarray(DAPI_arr, dtype=float)         

    """ Generate candidate masks ==> optional """
    scale = 0.227
    DAPI_size = round(18 / scale);   # calculate DAPI_size in pixels from scale (0.6904) and 18 um^2 average cell size
    DAPI_tmp,  total_matched_DAPI, total_DAPI = pre_process_QL(input_arr_tmp, counter[i], DAPI_size, name=onlyfiles_mask[counter[i]])
    
  
    labelled = measure.label(DAPI_tmp)
    cc = measure.regionprops(labelled)
    
    """ Initiate list of MACHINE counted CELL OBJECTS """
    num_MBP_pos = len(cc)
    list_M_cells = []
    for T in range(num_MBP_pos):
        cell = Cell(T)
        list_M_cells.append(cell)
        
    """ start looping and cropping """
    N = 0
    table_results = []
    seg_im = np.zeros(np.shape(DAPI_tmp))
    overlap_im = np.zeros(np.shape(DAPI_tmp))
    DAPI_im = np.zeros(np.shape(DAPI_tmp))

    while N < len(cc):  
        #batch_x = []; batch_y = [];
        DAPI_idx = cc[N]['centroid']
        
        # extract CROP outo of everything     
        input_crop, coords = adapt_crop_DAPI(input_arr, DAPI_idx, length=len_im, width=width_im)
     
        """ Create empty image with ONLY the DAPI at the DAPI_idx """
        DAPI_coords = cc[N]['coords']
        tmp = np.zeros(DAPI.shape)

        for T in range(len(DAPI_coords)):
            tmp[DAPI_coords[T,0], DAPI_coords[T,1]] = 255    
        
        tmp = Image.fromarray(tmp)
        DAPI_crop, coords = adapt_crop_DAPI(tmp, DAPI_idx, length=len_im, width=width_im)
            
        #DAPI_crop, coords = adapt_crop_DAPI_ARRAY(tmp, DAPI_idx, length=len_im, width=width_im)          

  
        """ENSURE COORDINATES WITHIN correct size"""
        size = np.shape(input_crop)
        width = size[0]
        length = size[1]
        
        while True:
            print('bad crop')
            c_width = int(coords[1]) - int(coords[0])
            c_length = int(coords[3]) - int(coords[2])
            if c_width > width:      coords[1] = coords[1] - 1
            elif c_width < width:    
                coords[1] = coords[1] + 1
                if coords[1] > width:  # in case it goest out of bounds
                    coords[0] = coords[0] - 1
                
            if c_length > length:    coords[3] = coords[3] - 1
            elif c_length < length:  
                coords[3] = coords[3] + 1
                if coords[3] > length:
                    coords[2] = coords[2] - 1
                    
            if c_width  == width and c_length == length:
                break;
                
        """ Delete green channel by making it the DAPI_mask instead
        """
        input_crop[:, :, 1] = DAPI_crop

        """ Normalize the image first """
        input_crop = normalize_im(input_crop, mean_arr, std_arr)  
       
        truth_im = np.zeros([len_im, width_im, 2])
        """ set inputs and truth """
        batch_x.append(input_crop)
        batch_y.append(truth_im) 
        
        feed_dict = {x:batch_x, y_:batch_y, training:0}      

        """ FEED_INPUT to NETWORK """
        output = softMaxed.eval(feed_dict=feed_dict)
        classification = np.argmax(output, axis = -1)[0]
        
        """ Plot for debug """       
#        plt.figure('Out'); plt.clf; plt.subplot(224); show_norm(input_crop[:, :, 0:3]); plt.pause(0.05); 
#        plt.subplot(221); 
#        true_m = np.argmax((batch_y[0]).astype('uint8'), axis=-1); plt.imshow(true_m);       
#        plt.title('Truth');
#        plt.subplot(222); plt.imshow(DAPI_crop); plt.title('DAPI_mask');
#        plt.subplot(223); plt.imshow(classification); plt.title('Output_seg');
#        plt.pause(0.05); 
        
        
        """ Skeletonize and count number of fibers within the output ==> saved for "sticky-seperate" later """
        copy_class = np.copy(classification)
        skel = skel_one(copy_class, minLengthDuring)
        labelled = measure.label(skel)
        cc_overlap = measure.regionprops(labelled)  
    
        for T in range(len(cc_overlap)):
            length = cc_overlap[T]['MajorAxisLength']
            angle = cc_overlap[T]['Orientation']
            overlap_coords = cc_overlap[T]['coords']
            
            if length > minLengthDuring and (angle > +0.785398 or angle < -0.785398):
                cell_num = N
                list_M_cells[cell_num].add_fiber(length)       
                                
                add_coords = [int(coords[0]), int(coords[2])]
                overlap_coords = overlap_coords + add_coords
                list_M_cells[cell_num].add_coords(overlap_coords)  
 

                
        """ Create mask of all segmented cells, also save as table """
        classification[classification > 0] = N + 1
        cropped_seg = seg_im[int(coords[0]): int(coords[1]), int(coords[2]): int(coords[3])]          
        added_seg = cropped_seg + classification + DAPI_crop
        seg_im[int(coords[0]): int(coords[1]), int(coords[2]): int(coords[3])] = added_seg  

        """ Create mask of OVERLAPPED, by just adding ones together """
        classification[classification > 0] = 1
        cropped_seg = overlap_im[int(coords[0]): int(coords[1]), int(coords[2]): int(coords[3])]          
        added_seg = cropped_seg + classification
        overlap_im[int(coords[0]): int(coords[1]), int(coords[2]): int(coords[3])] = added_seg  

        """ Create mask of DAPI, by just adding ones together """        
        DAPI_crop[DAPI_crop > 0] = N + 1
        cropped_seg = DAPI_im[int(coords[0]): int(coords[1]), int(coords[2]): int(coords[3])]          
        added_seg = cropped_seg + DAPI_crop
        DAPI_im[int(coords[0]): int(coords[1]), int(coords[2]): int(coords[3])] = added_seg 
        batch_x = []; batch_y = []      
        total_counter = total_counter + 1      
        N = N + 1
        print('Tested: %d of total: %d' %(total_counter, len(cc)))
   
    
    """ Get mask of regions that have overlap """
    binary_overlap = overlap_im > 0
    labelled = measure.label(binary_overlap)
    cc_overlap = measure.regionprops(labelled, intensity_image=overlap_im)
    
    masked = np.zeros(seg_im.shape)
    no_overlap = np.zeros(seg_im.shape)
    for M in range(len(cc_overlap)):
        overlap_val = cc_overlap[M]['MaxIntensity']
        overlap_coords = cc_overlap[M]['coords']
        if overlap_val > 1:    # if there is overlap
            for T in range(len(overlap_coords)):
                masked[overlap_coords[T,0], overlap_coords[T,1]] = seg_im[overlap_coords[T,0], overlap_coords[T,1]]
        else:  # no overlap
            for T in range(len(overlap_coords)):
                no_overlap[overlap_coords[T,0], overlap_coords[T,1]] = seg_im[overlap_coords[T,0], overlap_coords[T,1]]        
            
    import copy
    copy_list = copy.deepcopy(list_M_cells)
    
    """ associate fibers to masked """
    sort_mask = sort_max_fibers(masked, list_M_cells)

    """ Then add to all the cells that have no_overlap """
    all_fibers = np.add(no_overlap, sort_mask)

    """ Initiate list of CELL OBJECTS """
    num_MBP_pos = N
    list_cells = []
    for Q in range(N + 1):
        cell = Cell(N)
        list_cells.append(cell)
            
    """ Eliminate anything smaller than minLength, and in wrong orientation, then add to cell object """
    binary_all_fibers = all_fibers > 0
    labelled = measure.label(binary_all_fibers)
    cc_overlap = measure.regionprops(labelled, intensity_image=all_fibers)
    
    final_counted = np.zeros(all_fibers.shape)
    for Q in range(len(cc_overlap)):
        length = cc_overlap[Q]['MajorAxisLength']
        angle = cc_overlap[Q]['Orientation']
        overlap_coords = cc_overlap[Q]['coords']
        if length > minLength and (angle > +0.785398 or angle < -0.785398):
            #print(angle)
            cell_num = cc_overlap[Q]['MinIntensity']
            cell_num = int(cell_num) 
            
            """ PROBLEM, for some reason still some intensity values that go BEYOND the area
                maybe just switch above from Max to Min intensity?
            """                
            if cell_num > N:
                continue
            list_cells[cell_num].add_fiber(length)
    
            for T in range(len(overlap_coords)):
                final_counted[overlap_coords[T,0], overlap_coords[T,1]] = cell_num

    """ Print text onto image """
    filename_split = filename.split('.')[0]
    output_name = sav_dir + 'all_fibers_image' + '_' + filename_split + '_' + str(i) + '.png'
    add_text_to_image(final_counted, filename=output_name)             

    """ Garbage collection """
    DAPI_arr = []; DAPI_im = []; binary_overlap = []; labelled = []; overlap_im = []; seg_im = [];
    final_counted = []; sort_mask = []; tmp = []; binary_all_fibers = [];
    masked = []; no_overlap = [];
    DAPI_crop = []; added_seg = []; classification = []; copy_class = []; cropped_seg = []; input_crop = [];
    output = []; skel = []; truth_im = [];

    """ SAVING """
    with open(sav_dir + 'all_fibers' + '_' + filename_split + '_' + str(i) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
       pickle.dump([all_fibers], f) 
    
    """ Skeletonize and output
    """
    copy_all_fibers = np.copy(all_fibers)

    new_fibers = skeletonize_all_fibers(copy_all_fibers, i, DAPI_tmp=np.zeros(np.shape(all_fibers)), minLength=minLength,
                                         total_DAPI=total_DAPI, total_matched_DAPI=total_matched_DAPI,
                                         minLengthSingle=minSingle, s_path=sav_dir, name=filename_split)
    
    input_save = np.copy(np.asarray(input_arr))
    new_fibers[new_fibers > 0] = 255
    input_save[:,:,1] = new_fibers
    plt.imsave(sav_dir + 'final_image' + '_' + filename_split + '_' + str(i) + '.tif', (input_save))
    # garbage collection
    copy_all_fibers = []; 
    input_arr = []; 

    
def validation():

    # for validation
    val_input_path='C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Validation/'
    
    #input_path = 'E:/Tiger/UNet_new_data_1000px/'
    
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
    for T in range(len(onlyfiles_val)):
        """ Get validation images """
        input_val, truth_val = load_training(val_input_path, onlyfiles_val[T])

        """ Normalize the image first """
        input_val = normalize_im(input_val, mean_arr, std_arr) 
        
        batch_x_val.append(input_val)
        batch_y_val.append(truth_val)     
        #plt.figure(); plt.imshow(input_val)

    train_img[0] = batch_x_val[1]
                      
    true_img = list(range(1))
    true_img[0] = batch_y_val[1]

    true_weights = list(range(1))
    true_weights[0] = batch_weights_val[1]
                     
    feed_dict = {x:train_img,y_:true_img, weight_matrix:true_weights}

    """ Plots output of each filter layer """              
    plotLayers(feed_dict, L1, L2, L3, L4, L5, L6, L8, L9, L11)
    
    # PLOT THE OUTPUT CLASSIFICATION:
    classification = np.argmax(softMaxed.eval(feed_dict=feed_dict), axis = -1)[0]
    plt.figure('Fully Convolutional Network'); plt.clf();
    plt.subplot(231); plt.imshow(classification, vmin = 0, vmax = 1); plt.title('Segmentation');
    plt.subplot(232); show_norm(train_img[0][:, :, 0:3])
    plt.subplot(234); true_m = np.argmax((true_img[0]).astype('uint8'), axis=-1)  
    plt.imshow(true_m, vmin = 0, vmax = 1); plt.title('Truth_Mask');
                  
    plt.subplot(235); plt.imshow(train_img[0][:, :, 1]); plt.ylabel('DAPI');
    
    
    """ Training loss"""
    loss_t = cross_entropy.eval(feed_dict=feed_dict);
    loss_t
    
    """ Training loss"""
    jaccard_t = jaccard.eval(feed_dict=feed_dict);
    jaccard_t
    
