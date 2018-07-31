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



def run_analysis(s_path, sav_dir, input_path, checkpoint,
                 im_scale, minLength, minSingle, minLengthDuring, radius,
                 len_x, width_x, channels, CLAHE,
                 debug):
    
    # Variable Declaration
    x = tf.placeholder('float32', shape=[None, len_x, width_x, channels], name='InputImage')
    y_ = tf.placeholder('float32', shape=[None, len_x, width_x, 2], name='CorrectLabel')
    training = tf.placeholder(tf.bool, name='training')
    """ Creates network """
    y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed = create_network(x, y_, training)
    sess = tf.InteractiveSession()
    """ TO LOAD OLD CHECKPOINT """
    saver = tf.train.Saver()
    saver.restore(sess, s_path + 'check_' + checkpoint)
    
    
    """ Pre-processing """
    # Read in file names
    onlyfiles_mask = [ f for f in listdir(input_path) if isfile(join(input_path,f))]   
    natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
    onlyfiles_mask.sort(key = natsort_key1)
    
    counter = list(range(len(onlyfiles_mask)))  # create a counter, so can randomize it
    
    """ Load avg_img and std_img from TRAINING SET """
    mean_arr = 0; std_arr = 0;
    with open('mean_arr.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        mean_arr = loaded[0]
             
    with open('std_arr.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        std_arr = loaded[0]
  
    # Read in file names
    onlyfiles_mask = [ f for f in listdir(input_path) if isfile(join(input_path,f))]   
    natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
    onlyfiles_mask.sort(key = natsort_key1)
    
    counter = list(range(len(onlyfiles_mask)))  # create a counter, so can randomize it

    batch_x = []; batch_y = [];
    for i in range(len(onlyfiles_mask)):  
        total_counter = 0
        filename = onlyfiles_mask[counter[i]]
        if filename.split('.')[-1] != 'tif':
            continue
        
        """ Load image """
        #size = 3788 # 4775 and 6157 for the newest one
        input_arr = readIm_counter(input_path,onlyfiles_mask, counter[i]) 
        size_whole = input_arr.size[0]
        #size = int((size_whole * im_scale) / 0.227) # 4775 and 6157 for the newest one
        size = int((size_whole)) # 4775 and 6157 for the newest one
        input_arr = resize_adaptive(input_arr, size, method=Image.BICUBIC)
        size_whole = input_arr.size
        """ DO CLAHE """
        if CLAHE == 1:
            input_arr = exposure.equalize_adapthist(np.asarray(input_arr, dtype='uint8'), kernel_size=None, clip_limit=0.01, nbins=256)
            input_arr = np.asarray(input_arr * 255, dtype='uint8')
            input_arr = Image.fromarray(input_arr)
        
        DAPI_size = round(radius * radius * math.pi);  
        DAPI_tmp, total_matched_DAPI, total_DAPI = pre_process(input_arr, counter[i], DAPI_size, name=onlyfiles_mask[counter[i]], sav_dir=sav_dir)
        
        labelled = measure.label(DAPI_tmp)
        cc = measure.regionprops(labelled)
        
        """ Initiate list of MACHINE counted CELL OBJECTS """
        num_MBP_pos = len(cc)
        list_M_cells = []
        for T in range(num_MBP_pos):
            cell = Cell(T)
            list_M_cells.append(cell)
            
        """ start looping and cropping """
        N = 0  # SKIP THE FIRST DAPI POINT b/c it's the background?
        table_results = []
        seg_im = np.zeros(np.shape(DAPI_tmp))
        overlap_im = np.zeros(np.shape(DAPI_tmp))
        DAPI_im = np.zeros(np.shape(DAPI_tmp))
    
        if np.shape(DAPI_tmp)[0] < 1024 or np.shape(DAPI_tmp)[1] < 640:
            seg_im = np.zeros([1024, 640])
            overlap_im = np.zeros([1024, 640])
            DAPI_im = np.zeros([1024, 640])        
                
        while N < len(cc):
            DAPI_idx = cc[N]['centroid']
            
            # extract CROP outo of everything          
            input_crop, coords = adapt_crop_DAPI(input_arr, DAPI_idx, length=len_x, width=width_x)
         
            """ Create empty image with ONLY the DAPI at the DAPI_idx """
            DAPI_coords = cc[N]['coords']
            tmp = np.zeros(np.shape(DAPI_tmp))
    
            for T in range(len(DAPI_coords)):
                tmp[DAPI_coords[T,0], DAPI_coords[T,1]] = 255
            
            tmp = Image.fromarray(tmp)
            DAPI_crop, coords = adapt_crop_DAPI(tmp, DAPI_idx, length=len_x, width=width_x)
    
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
            input_crop[:, :, 1] = np.zeros([len_x,width_x])       
            if channels == 4:
                tmp = np.zeros([len_x,width_x,channels])
                tmp[:,:,0:3] = input_crop
                tmp[:,:,3] = DAPI_crop
                input_crop = tmp
            elif channels == 3:
                input_crop[:,:,1] = DAPI_crop
    
    
            """ Normalize the image first """
            input_crop = normalize_im(input_crop, mean_arr, std_arr)  
           
            truth_im = np.zeros([len_x, width_x, 2])
            """ set inputs and truth """
            batch_x.append(input_crop)
            batch_y.append(truth_im) 
            
            feed_dict = {x:batch_x, y_:batch_y, training:0}      
    
            """ FEED_INPUT to NETWORK """
            output = softMaxed.eval(feed_dict=feed_dict)
            classification = np.argmax(output, axis = -1)[0]
            
            """ Plot for debug """ 
            if debug:
                plt.figure('Out'); plt.clf; plt.subplot(224); show_norm(input_crop[:, :, 0:3]); plt.pause(0.05); 
                plt.subplot(221); 
                true_m = np.argmax((batch_y[0]).astype('uint8'), axis=-1); plt.imshow(true_m);       
                plt.title('Truth');
                plt.subplot(222); plt.imshow(DAPI_crop); plt.title('DAPI_mask');
                plt.subplot(223); plt.imshow(classification); plt.title('Output_seg');
                plt.pause(0.05); 
                
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
        new_fibers = skeletonize_all_fibers(copy_all_fibers, i, DAPI_tmp=np.zeros([size_whole[0],size_whole[1]]), minLength=minLength,
                                             total_DAPI=total_DAPI, total_matched_DAPI=total_matched_DAPI,
                                             minLengthSingle=minSingle, s_path=sav_dir, name=filename_split)
            
        input_save = np.copy(np.asarray(input_arr))
        
        new_fibers[new_fibers > 0] = 255
        input_save[0:np.minimum(input_arr.size[0], new_fibers.shape[1]), 0:np.minimum(input_arr.size[1], new_fibers.shape[0]),1] = new_fibers[0:np.minimum(input_arr.size[0], new_fibers.shape[1]), 0:np.minimum(input_arr.size[1], new_fibers.shape[0])]
        plt.imsave(sav_dir + 'final_image' + '_' + filename_split + '_' + str(i) + '.tif', (input_save))
        # garbage collection
        
        copy_all_fibers = []; 
        input_arr = []; 
        
    sess.close()