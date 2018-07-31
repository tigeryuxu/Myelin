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

""" Network Begins:
"""
s_path = 'C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Checkpoints/Check_2_noDrop_online/'
s_path = 'C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Checkpoints/Check_MyQ7_noDrop_small/'
s_path = 'C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Checkpoints/Check_MyQz11_sW/'

input_path = 'C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Testing/Input/'
DAPI_path='C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Testing/Candidates/'


""" Load avg_img and std_img from TRAINING SET """
mean_arr = 0; std_arr = 0;
with open('mean_arr.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    loaded = pickle.load(f)
    mean_arr = loaded[0]
         
with open('std_arr.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    loaded = pickle.load(f)
    std_arr = loaded[0]

# Variable Declaration
x = tf.placeholder('float32', shape=[None, 1024, 640, 3], name='InputImage')
y_ = tf.placeholder('float32', shape=[None, 1024, 640, 2], name='CorrectLabel')
training = tf.placeholder(tf.bool, name='training')

# Read in file names
onlyfiles_mask = [ f for f in listdir(input_path) if isfile(join(input_path,f))]   
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
onlyfiles_mask.sort(key = natsort_key1)

# Read in file names
onlyfiles_DAPI = [ f for f in listdir(DAPI_path) if isfile(join(DAPI_path,f))]
onlyfiles_DAPI.sort(key = natsort_key1)

counter = list(range(len(onlyfiles_mask)))  # create a counter, so can randomize it

""" Creates network and cost function"""
y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed = create_network(x, y_, training)
accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm_CLASSW(y, y_b, logits)

sess = tf.InteractiveSession()
""" TO LOAD OLD CHECKPOINT """
saver = tf.train.Saver()
saver.restore(sess, s_path + 'check_222000')


batch_x = []; batch_y = [];
for i in range(len(onlyfiles_mask)):  
    total_counter = 0
    filename = onlyfiles_mask[counter[i]]
    
    """ Load image
    """
    input_arr = readIm_counter(input_path,onlyfiles_mask, counter[i]) 
    DAPI_arr = readIm_counter(DAPI_path,onlyfiles_DAPI, counter[i])
    DAPI_tmp = np.asarray(DAPI_arr, dtype=float)        
  
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
        batch_x = []; batch_y = [];
        DAPI_idx = cc[N]['centroid']
        
        # extract CROP outo of everything          
        input_crop, coords = adapt_crop_DAPI(input_arr, DAPI_idx, length=1024, width=640)
     
        """ Create empty image with ONLY the DAPI at the DAPI_idx """
        DAPI_coords = cc[N]['coords']
        tmp = np.zeros(DAPI_arr.size)

        for T in range(len(DAPI_coords)):
            tmp[DAPI_coords[T,0], DAPI_coords[T,1]] = 255    
        DAPI_crop, coords = adapt_crop_DAPI_ARRAY(tmp, DAPI_idx, length=1024, width=640)          

  
        """ENSURE COORDINATES WITHIN correct size"""
        size = np.shape(input_crop)
        length = size[1]
        width = size[0]
        c_width = int(coords[1]) - int(coords[0])
        c_length = int(coords[3]) - int(coords[2])
        
        if c_width > width:      coords[1] = coords[1] - 1
        elif c_width < width:    
            coords[1] = coords[1] + 1
            if coords[1] > 8208:  # in case it goest out of bounds
                coords[0] = coords[0] - 1
            
        if c_length > length:    coords[3] = coords[3] - 1
        elif c_length < length:  
            coords[3] = coords[3] + 1
            if coords[3] > 8208:
                coords[2] = coords[2] - 1
                
        """ Delete green channel by making it the DAPI_mask instead
        """
        input_crop[:, :, 1] = DAPI_crop

        """ Normalize the image first """
        input_crop = normalize_im(input_crop, mean_arr, std_arr)  
       
        truth_im = np.zeros([1024, 640, 2])
        """ set inputs and truth """
        batch_x.append(input_crop)
        batch_y.append(truth_im) 
        
        feed_dict = {x:batch_x, y_:batch_y, training:0}      

        """ FEED_INPUT to NETWORK """
        output = softMaxed.eval(feed_dict=feed_dict)
        classification = np.argmax(output, axis = -1)[0]
        

        """ Skeletonize and count number of fibers within the output ==> saved for "sticky-seperate" later """
        minLength = 10;
        copy_class = np.copy(classification)
        skel = skel_one(copy_class, minLength)
        labelled = measure.label(skel)
        cc_overlap = measure.regionprops(labelled)  
    
        for T in range(len(cc_overlap)):
            length = cc_overlap[T]['MajorAxisLength']
            angle = cc_overlap[T]['Orientation']
            overlap_coords = cc_overlap[T]['coords']
            
            if length > minLength and (angle > +0.785398 or angle < -0.785398):
                cell_num = N
                list_M_cells[cell_num].add_fiber(length)       
                                
                add_coords = [int(coords[0]), int(coords[2])]
                overlap_coords = overlap_coords + add_coords
                list_M_cells[cell_num].add_coords(overlap_coords)  

                
        """ Create mask of all segmented cells, also save as table """
        classification[classification > 0] = N
        cropped_seg = seg_im[int(coords[0]): int(coords[1]), int(coords[2]): int(coords[3])]          
        added_seg = cropped_seg + classification + DAPI_crop
        seg_im[int(coords[0]): int(coords[1]), int(coords[2]): int(coords[3])] = added_seg  

        """ Create mask of OVERLAPPED, by just adding ones together """
        classification[classification > 0] = 1
        cropped_seg = overlap_im[int(coords[0]): int(coords[1]), int(coords[2]): int(coords[3])]          
        added_seg = cropped_seg + classification
        overlap_im[int(coords[0]): int(coords[1]), int(coords[2]): int(coords[3])] = added_seg  

        """ Create mask of DAPI, by just adding ones together """        
        DAPI_crop[DAPI_crop > 0] = N
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
    for Q in range(N):
        cell = Cell(N)
        list_cells.append(cell)
            
    """ Eliminate anything smaller than minLength, and in wrong orientation, then add to cell object """
    minLength = 25
    binary_all_fibers = all_fibers > 0
    labelled = measure.label(binary_all_fibers)
    cc_overlap = measure.regionprops(labelled, intensity_image=all_fibers)
    
    final_counted = np.zeros(all_fibers.shape)
    for Q in range(len(cc_overlap)):
        length = cc_overlap[Q]['MajorAxisLength']
        angle = cc_overlap[Q]['Orientation']
        overlap_coords = cc_overlap[Q]['coords']
        length = cc_overlap[T]['MajorAxisLength']
    
        #print(angle)
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
        
    """ SAVING """
    with open('all_fibers' + str(i) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
       pickle.dump([all_fibers], f)      

    """ Garbage collection """
    DAPI_arr = []; DAPI_im = []; binary_overlap = []; input_arr = []; labelled = []; overlap_im = []; seg_im = [];
    final_counted = []; sort_mask = []; tmp = []; binary_all_fibers = [];
    masked = []; no_overlap = [];
    DAPI_crop = []; added_seg = []; classification = []; copy_class = []; cropped_seg = []; input_crop = [];
    output = []; skel = []; truth_im = [];
    
    """ Skeletonize and output
    """
    copy_all_fibers = np.copy(all_fibers)
    skeletonize_all_fibers(copy_all_fibers, i, DAPI_tmp=np.zeros([8208,8208]), minLength=25, minLengthSingle=150)
        
    # garbage collection
    copy_all_fibers = []; 



 """ PRINT NUMBERS OVER THE IMAGE """
add_text_to_image(final_fibers, filename='default.png')
def add_text_to_image(all_fibers, filename='default.png'):
    import random
    fiber_img = Image.fromarray((all_fibers *255).astype(np.uint16))
    plt.figure(figsize=(12,10)); plt.imshow(fiber_img)
    plt.axis('off')
    # PRINT TEXT ONTO IMAGE
    binary_all_fibers = all_fibers > 0
    labelled = measure.label(binary_all_fibers)
    cc_overlap = measure.regionprops(labelled, intensity_image=all_fibers)
    
    final_counted = np.zeros(all_fibers.shape)
    cell_num = 0
    
    # Make a list of random colors corresponding to all the cells
    list_fibers = []
    for Q in range(int(np.max(all_fibers) + 1)):
        color = [random.randint(0,255)/256, random.randint(0,255)/256, random.randint(0,255)/256]
        list_fibers.append(color)
        
    for Q in range(len(cc_overlap)):
        length = cc_overlap[Q]['MajorAxisLength']
        angle = cc_overlap[Q]['Orientation']
        overlap_coords = cc_overlap[Q]['coords']
        new_num = cc_overlap[Q]['MinIntensity']
        
        #if cell_num != new_num:
            #color = [random.randint(0,255)/256, random.randint(0,255)/256, random.randint(0,255)/256]
            #cell_num = new_num
        color = list_fibers[int(new_num)]
        plt.text(overlap_coords[0][1], overlap_coords[0][0], str(int(new_num)), fontsize= 2, color=color)
    
    
    plt.savefig(filename, dpi = 500)
    
    
    
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

    train_img = list(range(1))
    train_img[0] = batch_x_val[1]
                      
    true_img = list(range(1))
    true_img[0] = batch_y_val[1]
                     
    feed_dict = {x:train_img,y_:true_img}

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
    
    
    
for i in range(0, len(plot_jaccard),20):
    print(plot_jaccard[i])
    print(i)
    
""" 3rd try 236000
1st == 0.58911985
2nd == 0.16671689
"""

""" 3rd try 275000
1st == 0.5812
2nd == 0.0719
"""