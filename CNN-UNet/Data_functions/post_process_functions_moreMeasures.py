# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:25:15 2017

@author: Tiger

*** DARYAN IS MODIFYING IT TO ADD MEASUREMENTS
- adding new methods to cell class to store new measurements
- these are m

*** MEASUREMENTS ADDED:
    - morphological
        - xCent = x coordinate of centroid
        - yCent = y coordinate of centroid
    - intensity
        - int = mean intensity
        - var = st deviation
        - min = min intensity
        - max = max intensity
        
*** TO-DOs
    - add same intensity measurements in green channel
    - same intensity measurements PLUS circularity of DAPI ROI
    - "inter-cell" measurements, ie proximity to other sheaths, density of 
        non-ensheathing cells nearby
    - rewrite and simplify some things - mainly just have skeletonize_all_fibers
        output masked all_fibers (so you have the labels) and then use this
        with regionprops to get all measurements in one function
"""

from skimage.morphology import skeletonize
from skimage.morphology import *
from skimage import data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.util import invert
import mahotas as mah
import cv2
import numpy as np
from skimage import measure
import csv
from PIL import Image
from os import listdir
import pickle as pickle
import os

import numpy.ma as ma
import pandas as pd

from Data_functions.data_functions import *
from Data_functions.plot_functions import *
from Data_functions.UNet import *


""" defines a cell object for saving output """
class Cell:
    def __init__(self, num):
        self.num = num   
        
        self.fibers = []   # list of lengths
        self.xCent = []     # list of x coordinates of centroids
        self.yCent = []
        
        self.intS = []
        self.varS = []
        self.minS = []
        self.maxS = []
        
        self.intE = []
        self.varE = []
        self.minE = []
        self.maxE = []
        
        self.intB = []  # body measures are int not lists (obvi only 1 body per cell)
        self.varB = []
        self.minB = []
        self.maxB = []
        
        self.centB = [] # tuple with x and y coord (ultimately makes more sense than separating as I did for sheaths)
        self.areaB = []
        self.circB = []
        
        self.coords = np.zeros([1, 2], dtype=int)

    def add_fiber(self, fibers):   # where "fibers" is a single value length
        self.fibers.append(fibers)

    def add_coords(self, new_coords):
        self.coords = np.append(self.coords, new_coords, axis=0)
        
    def add_xCent(self, xCent): # X and Y centroid coordinates
        self.xCent.append(xCent)
        
    def add_yCent(self, yCent):
        self.yCent.append(yCent)  
        
    def add_intS(self, int):  # sheath-channel mean intensity
        self.intS.append(int)
        
    def add_varS(self, var):  # sheath-channel intensity
        self.varS.append(var)      
        
    def add_minS(self, min):  # sheath-channel intensity
        self.minS.append(min)
        
    def add_maxS(self, max):  # sheath-channel intensity
        self.maxS.append(max)        


    def add_intE(self, intE):  # extra-channel intensity in sheaths
        self.intE.append(intE)
        
    def add_varE(self, var): 
        self.varE.append(var)      
        
    def add_minE(self, min):  
        self.minE.append(min)
        
    def add_maxE(self, max):
        self.maxE.append(max)        

   
    def add_intB(self, intB):  # sheath-channel intensity in cell body
        self.intB.append(intB)
        
    def add_varS(self, var):  
        self.varS.append(var)      
        
    def add_minS(self, min): 
        self.minS.append(min)
        
    def add_maxS(self, max): 
        self.maxS.append(max)        

    def add_centB(self, centB): # soma centroid
        self.centB.append(centB)
        
    def add_areaB(self, area): # soma area
        self.areaB.append(area)
        
    def add_circB(self, circ): # soma circularity
        self.circB.append(circ)
        
        
   
   
""" Instead of getting minimum intensity:
        1) loop through all of the masked overlapping regions
          and find out ALL of the UNIQUE values within the NON-overlapped regions
        2) use these unqique values to index the list of cells
        3) find the number of fibers associated with each of these cells, and assign it to the cell with the MOST 
           number of fibers ALREADY associated and that has AT LEAST more than 1 fiber???
           if all only have 1 fiber, then just assign to the one that has the minium intensity???
"""
def sort_max_fibers(masked, list_M_cells):
    """ maybe add a step where you cycle through and get all the indexes of cells WITH fibers
        so don't have to waste time later looping over cells that don't even have fibers        
    """
    idx_cells = [] 
    for T in range(len(list_M_cells)):
        fibers = list_M_cells[T].fibers
        if fibers:
            idx_cells.append(T)
    
    import operator    
    binary_masked = masked > 0
    labelled = measure.label(binary_masked)
    cc_overlap = measure.regionprops(labelled, intensity_image=masked)
    sort_mask = np.zeros(masked.shape)

    for M in range(len(cc_overlap)): 
        
        overlap_coords = cc_overlap[M]['coords']
        
        cells_overlap = []
        all_numFibers = []
        for T in range(len(idx_cells)):   
           idx = idx_cells[T]
                  
           fiber_coords = list_M_cells[idx].coords 
           fibers = list_M_cells[idx].fibers 
           
           combined = np.append(overlap_coords, fiber_coords, axis=0)
           orig_len = len(combined)
           
           """ find if fiber overlaps by seeing if there is anything unique 
               RATHER than actually seeing if every pixel matches
           """
           uniq = np.unique(combined, axis=0)
           if len(uniq) < orig_len:
               cells_overlap.append(idx)
               all_numFibers.append(len(fibers))                      
                          
        if len(cells_overlap) > 1:
            cell_index, value = max(enumerate(all_numFibers), key=operator.itemgetter(1))
            
            """ (4) set the entire region to be of value cell_index """    
            for T in range(len(overlap_coords)):
               sort_mask[overlap_coords[T,0], overlap_coords[T,1]] = cells_overlap[cell_index]        
        
        print('Tested: %d overlapped of total: %d' %(M, len(cc_overlap)))     

    return sort_mask


"""
Find branch point in example image.
"""
def find_branch_points(sk):
    branch1=np.array([[2, 1, 2], [1, 1, 1], [2, 2, 2]])
    branch2=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch3=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]])
    branch4=np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]])
    branch5=np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
    branch6=np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]])
    branch7=np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch8=np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]])
    branch9=np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
    br1=mah.morph.hitmiss(sk,branch1)
    br2=mah.morph.hitmiss(sk,branch2)
    br3=mah.morph.hitmiss(sk,branch3)
    br4=mah.morph.hitmiss(sk,branch4)
    br5=mah.morph.hitmiss(sk,branch5)
    br6=mah.morph.hitmiss(sk,branch6)
    br7=mah.morph.hitmiss(sk,branch7)
    br8=mah.morph.hitmiss(sk,branch8)
    br9=mah.morph.hitmiss(sk,branch9)    
    br=br1+br2+br3+br4+br5+br6+br7+br8+br9
    return br


""" re-runs all the outputs to modify the fiber threshold"""
def rerun_all():    
    input_path = './SPATIAL_W_301000_Laminin_PDL/'
    all_csv = read_file_names(input_path)   
    for T in range(len(all_csv)):
        all_fibers = load_pkl(input_path, all_csv[T])
        
        if T < 5:
            add = 11
        elif T < 10: 
            add = 21 - 5

        skeletonize_all_fibers(all_fibers, T + add, DAPI_tmp = np.zeros([8208,8208]), minLength=18, minLengthSingle=72)    



""" Read and combine csv into single files containing lengths, numsheaths, ect...
"""
def read_and_comb_csv_as_SINGLES(input_path):
    all_fibers = []
    all_numCells = []
    all_numShea = []
    all_numMFLC = []
    
    filenames = listdir(input_path)
    all_csv  = [filename for filename in filenames if filename.endswith(".csv") ]
    first = 1;
    output_name = all_csv[0] 
    output_name = output_name.split('.')[0]
    
    directory = input_path + 'combined_CSVs/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + 'Results_' + output_name + '_num_sheaths.csv', 'w+',  newline='') as sheaths:
        with open(directory + 'Results_' +  output_name + '_lengths.csv', 'w+',  newline='') as lengths:
            with open(directory + 'Results_' + output_name + '_cells.csv', 'w+',  newline='') as cells:
               with open(directory + 'Results_' + output_name + '_mSLC.csv', 'w+',  newline='') as mFLC:

                    for T in range(len(all_csv)):
                        
                        filename = all_csv[T]
                        empty = 0
                        with open(input_path + filename, 'r') as csvfile:
                            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                            counter = 0
                        
                            for row in spamreader:
                                row = list(filter(lambda a: a != '[]', row))
                            
                                if counter % 2 != 0:
                                    counter = counter + 1
                                    continue
                                
                                for t in range(len(row)):
                                    if row[t] == '[]' or not row[t] :
                                        continue
                                    row[t] =  float(row[t])
                                    
                                if row == []:
                                   row = ['-']

                                if counter == 0:   all_fibers.append(row); wr = csv.writer(lengths, quoting=csv.QUOTE_ALL); wr.writerow(all_fibers[0]);
                                elif counter == 2: 
                                    all_numCells.append(row[0]);   # append the Num Ensheathed 
                                elif counter == 8:
                                    all_numCells.append(row[0]);   # append the Num MBP+
                                elif counter == 10:
                                    all_numCells.append(row[0]);   # append the Num Cells   
                                    wr = csv.writer(cells, quoting=csv.QUOTE_ALL); 
                                    wr.writerow(all_numCells);
                                    all_numCells = []
                                    
                                elif counter == 4: all_numShea.append(row); wr = csv.writer(sheaths, quoting=csv.QUOTE_ALL); wr.writerow(all_numShea[0]);
                                elif counter == 6: all_numMFLC.append(row); wr = csv.writer(mFLC, quoting=csv.QUOTE_ALL); wr.writerow(all_numMFLC[0]);

                                all_fibers = []
                                #all_numCells = []
                                all_numShea = []
                                all_numMFLC = []

                                if counter == 10:
                                    break
                                counter = counter + 1
                            
                        if not empty:
                            first = 0    


""" go through list_cells to get all the information """
def cycle_and_output_csv(list_cells, output_name, minLengthSingle, total_DAPI=0, total_matched_DAPI=0, s_path=''):
    num_wrap = 0
    wrap_per_cell = []
    all_fiber_lengths = []
    mFLC = []
    
    xCents = []
    yCents = []
    
    new_list = []
    for i in range(len(list_cells)):
        fibers = list_cells[i].fibers
        x = list_cells[i].xCent
        y = list_cells[i].yCent
                            
        if len(fibers) == 1 and fibers[0] < minLengthSingle:
            all_fiber_lengths.extend([]) 
            new_list.append([])
        elif len(fibers) == 2 and (fibers[0] < minLengthSingle and fibers[1] < minLengthSingle):
            all_fiber_lengths.extend([])
            new_list.append([])
        elif fibers:   # if it is NOT empty, then there are fibers
            num_wrap = num_wrap + 1
            wrap_per_cell.append(len(fibers))
            all_fiber_lengths.extend(fibers)
            mean = sum(fibers)/len(fibers)
            mFLC.append(mean)
            
            xCents.append(x)
            yCents.append(y)
            
            new_list.append(list_cells[i])
        else:
            all_fiber_lengths.extend([]) 
            new_list.append([])
    lis_props = [num_wrap]
    
    with open(s_path + output_name, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(all_fiber_lengths)
        wr.writerow(lis_props)
        wr.writerow(wrap_per_cell)
        wr.writerow(mFLC)
        wr.writerow([total_matched_DAPI])
        wr.writerow([total_DAPI])
        
        wr.writerow([xCents]) 
        wr.writerow([yCents])
        
    return new_list
        
        
""" NEW: find all that have width too large """
def width_separate(masked, all_fibers, width_thresh, minLength):
    binary_all_fibers = masked > 0
    labelled = measure.label(binary_all_fibers)
    cc_overlap = measure.regionprops(labelled, intensity_image=all_fibers)
    
    large_width = np.zeros(masked.shape)
    short_width = np.zeros(masked.shape)
    
    for M in range(len(cc_overlap)):
        length = cc_overlap[M]['MajorAxisLength']
        angle = cc_overlap[M]['Orientation']
        overlap_coords = cc_overlap[M]['coords']
        width = cc_overlap[M]['MinorAxisLength']
        
        if width > width_thresh and length > minLength and (angle > +0.785398 or angle < -0.785398):
            cell_num = cc_overlap[M]['MaxIntensity']
            cell_num = int(cell_num) 
    
            for T in range(len(overlap_coords)):
                large_width[overlap_coords[T,0], overlap_coords[T,1]] = cell_num

        else:
        
            for T in range(len(overlap_coords)):
                cell_num = cc_overlap[M]['MaxIntensity']
                cell_num = int(cell_num)      
            
                short_width[overlap_coords[T,0], overlap_coords[T,1]] = cell_num  
                
    """ CHANGED FROM 5,3 ==> 2,2 """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,3))
    dil = cv2.dilate(large_width,kernel,iterations = 1)
    kernel = np.ones((8,1),np.uint8)    
    opening = cv2.morphologyEx(dil, cv2.MORPH_OPEN, kernel)
    combined = np.add(opening, short_width)
    
    return combined



def skel_one(all_fibers, minLength):
    image = all_fibers
    image = image > 0
    skeleton = skeletonize(image)
    
    bp = find_branch_points(skeleton)
    
    """ Then dilate the branchpoints and subtract from the original image """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    bpd = cv2.dilate(bp,kernel,iterations = 1)
    bpd = bpd.astype(int)
    bpd[bpd > 0] = 1
    
    sub_im = skeleton - bpd
    sub_im[sub_im < 0] = 0
    sub_im[sub_im > 0] = 1
    
    """ Find EVERYTHING smaller than minLength, AND in wrong orientation, so can delete from whole image afterwards """
    smallLength = 0
    sub_im  = sub_im > 0
    labelled = measure.label(sub_im)
    cc_overlap = measure.regionprops(labelled)
    
    hor_lines = np.zeros(sub_im.shape)
    for i in range(len(cc_overlap)):
        length = cc_overlap[i]['MajorAxisLength']
        angle = cc_overlap[i]['Orientation']
        overlap_coords = cc_overlap[i]['coords']
        #print(angle)
        if length < smallLength or (angle <= +0.785398 and angle >= -0.785398):
    
            for T in range(len(overlap_coords)):
                hor_lines[overlap_coords[T,0], overlap_coords[T,1]] = 1
                    
    """ Then subtract the horizontal and too small lines from the original skeleton"""
    all_vert = skeleton - hor_lines                      
    
    """ then invert this, and use this as a mask over top of the "all_fibers" """    
    masked = all_fibers
    masked[all_vert == 0] = 0                

    """ SEPARATE BY WIDTH """
    width = 10
    combined = width_separate(masked, all_fibers, width, minLength)
    
    masked = all_fibers
    masked[combined == 0] = 0
    
    return masked
    
    
""" uses list_M_cells to find out where the DAPI nuclei are of cells with fibers """
def extract_ensheathed_DAPI(DAPI_tmp, list_cells):
    labelled = measure.label(DAPI_tmp)
    cc = measure.regionprops(labelled)

    DAPI_ensheathed = np.zeros(DAPI_tmp.shape)   
    num_cells = 0
    for i in range(len(list_cells)):
        fibers = list_cells[i].fibers
        if fibers and i < len(cc):
            overlap_coords = cc[i]['coords']
            for T in range(len(overlap_coords)):
                DAPI_ensheathed[overlap_coords[T,0], overlap_coords[T,1]] = i                
    
            num_cells = num_cells + 1
    
    return DAPI_ensheathed     


""" Takes an image and associates all the fibers in it to a list of cells"""
def fiber_to_list(masked, all_fibers, input_arr, list_cells, minLength):
        
    binary_all_fibers = masked > 0
    labelled = measure.label(binary_all_fibers)
    cc_overlap = measure.regionprops(labelled, intensity_image=all_fibers)
    
    imSheath = np.array(input_arr)[:,:,0] # MBP-only channel
   # imExtra = 
 
    final_counted = np.zeros(masked.shape)
    for M in range(len(cc_overlap)):
        cent = cc_overlap[M]['Centroid']
        xCent = cent[1]
        yCent = cent[0]
        
        length = cc_overlap[M]['MajorAxisLength']
        angle = cc_overlap[M]['Orientation']
        overlap_coords = cc_overlap[M]['coords']

        if length > minLength and (angle > +0.785398 or angle < -0.785398):
            cell_num = cc_overlap[M]['MaxIntensity']
            cell_num = int(cell_num) 

            list_cells[cell_num].add_fiber(length)
            list_cells[cell_num].add_coords(overlap_coords) 
            list_cells[cell_num].add_xCent(xCent)
            list_cells[cell_num].add_yCent(yCent)    
            sheaths = []
            # extras = []
            for T in range(len(overlap_coords)):        
                final_counted[overlap_coords[T,0], overlap_coords[T,1]] = cell_num #tiger added this for something don't touch
                c = overlap_coords[T]
                sheaths.append(imSheath[c[0],c[1]])
                # extras.append(imExtra[c[0],c[1]])
            sheath_arr = np.array(sheaths)  
            #extra_arr = np.array(extras)

            list_cells[cell_num].add_intS(int(sheath_arr.mean()))
            list_cells[cell_num].add_varS(float(sheath_arr.var()))
            list_cells[cell_num].add_maxS(float(sheath_arr.max()))
            list_cells[cell_num].add_minS(float(sheath_arr.min()))
            
            #list_cells[cell_num].add_intE(int(extra_arr.mean()))
            #list_cells[cell_num].add_varE(float(extra_arr.var()))
            #list_cells[cell_num].add_maxE(float(extra_arr.max()))
            #list_cells[cell_num].add_minE(float(extra_arr.min()))
            
            
    return list_cells, final_counted


""" Take final list and turn it into an image """
def im_from_list(list_cells, minLengthSingle, shape):

    new_fibers = np.zeros(shape)  
    num_fibers = 0
    for i in range(len(list_cells)):
        if list_cells[i]:
            fibers =  list_cells[i].fibers 
            if len(fibers) == 1 and fibers[0] < minLengthSingle:
                continue;
            elif len(fibers) == 2 and (fibers[0] < minLengthSingle and fibers[1] < minLengthSingle):
                continue;
            elif fibers:   # if it is NOT empty, then there are fibers
                coords = list_cells[i].coords
                for T in range(len(coords)):
                    new_fibers[coords[T,0], coords[T,1]] = i
                    
                num_fibers = num_fibers + 1
                    
    return new_fibers


""" Skeletonize and output final cell count """
def skeletonize_all_fibers(all_fibers, input_arr, i, DAPI_tmp, minLength, minLengthSingle, total_DAPI=0, total_matched_DAPI=0, s_path='', name='', jacc_test=0):

    im_num = i
    minLengthSingle = minLengthSingle
    #name = filename_split
    #s_path = sav_dir
    # Invert the image
    image = all_fibers
    image = image > 0
    
    """ different skeletonization methods to try out"""
    skeleton = skeletonize(image)        
    bp = find_branch_points(skeleton)
            
    """ Then dilate the branchpoints and subtract from the original image """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    bpd = cv2.dilate(bp,kernel,iterations = 1)
    bpd = bpd.astype(int)
    bpd[bpd > 0] = 1
    
    sub_im = skeleton - bpd
    sub_im[sub_im < 0] = 0
    sub_im[sub_im > 0] = 1
    
    """ Find EVERYTHING smaller than minLength, AND in wrong orientation, so can delete from whole image afterwards """
    smallLength = 0
    sub_im  = sub_im > 0
    labelled = measure.label(sub_im)
    cc_overlap = measure.regionprops(labelled)
    
    hor_lines = np.zeros(sub_im.shape)
    for i in range(len(cc_overlap)):
        length = cc_overlap[i]['MajorAxisLength']
        angle = cc_overlap[i]['Orientation']
        overlap_coords = cc_overlap[i]['coords']
        #print(angle)
        if length < smallLength or (angle <= +0.785398 and angle >= -0.785398):
    
            for T in range(len(overlap_coords)):
                hor_lines[overlap_coords[T,0], overlap_coords[T,1]] = 1
                     
    
    """ Then subtract the horizontal and too small lines from the original skeleton"""
    all_vert = skeleton - hor_lines      
        
    """ then invert this, and use this as a mask over top of the "all_fibers" """
    masked = np.copy(all_fibers)
    masked[all_vert == 0] = 0

    """ Clean garbage """
    all_vert = [];bp = []; bpd = []; hor_lines = []; image = []; labelled = []; skeleton = []; sub_im = [];
        
    """ SEPARATE BY WIDTH """
    width = 10
    combined = width_separate(masked, all_fibers, width, minLength)

    masked = np.copy(masked)
    masked[combined == 0] = 0

    """ Eliminate anything smaller than minLength, and in wrong orientation, then add to cell object """
    N = total_DAPI
        
    list_cells = []
    for M in range(N):
         cell = Cell(M)
         list_cells.append(cell)
    list_cells_sorted, final_counted = fiber_to_list(masked, all_fibers, input_arr, list_cells, minLength)    

#    """ Subtract expanded ensheathed DAPI spots b/c no fibers can pass through cell nucleus """
#    DAPI_ensheathed = extract_ensheathed_DAPI(DAPI_tmp, list_cells)
#
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
#    dilated_DAPI = cv2.dilate(DAPI_ensheathed,kernel,iterations = 1)
#
#    copy_final_counted = np.copy(final_counted)
#    copy_final_counted[dilated_DAPI > 0] = 0
#    
#    # clean GARBAGE
#    DAPI_ensheathed = []; dilated_DAPI = []; combined = [];
    
    list_cells = []
    for M in range(N):
         cell = Cell(M)
         list_cells.append(cell)
    list_cells_sorted, final_counted_new = fiber_to_list(final_counted, all_fibers, input_arr, list_cells, minLength)

    """ go through list_cells to get all the information """
    output_name = 'masked_out_dil' + '_' + name + '_' + str(im_num) + '.csv'
    new_list = cycle_and_output_csv(list_cells_sorted, output_name, minLengthSingle, total_DAPI, total_matched_DAPI, s_path=s_path)

    shape = np.shape(all_fibers)
    new_fibers = im_from_list(new_list, minLengthSingle, shape)
    #plt.imsave('final_image' + str(im_num) + '.tif', (new_fibers * 255).astype(np.uint16))
    
    sz = 5;
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz));   # get disk structuring element
    dil_final = cv2.dilate(new_fibers, kernel, 1)


    """ Print out pickles for jaccard testing """
    if jacc_test:   
        for_jaccard_testing(new_fibers, all_fibers, minLength, DAPI_tmp, im_num, N, s_path=s_path)

    
    return dil_final, new_fibers, new_list, list_cells
    

    
#""" FOR JACCARD TESTING """
#def for_jaccard_testing(new_fibers, all_fibers, minLength, DAPI_tmp, im_num, N, s_path):
#       
#    import pickle as pickle
#    """ Print text onto image """
#    #output_name = 'masked_out_dil' + str(im_num) + '.png'
#    #add_text_to_image(new_fibers, filename=output_name)
#           
#    """ Sort through the final DAPI ==> for Jaccard testing only"""
#    list_cells = []
#    for M in range(N):
#         cell = Cell(N)
#         list_cells.append(cell)
#    list_cells_sorted, final_counted_new = fiber_to_list(new_fibers, all_fibers, list_cells, minLength)
#    DAPI_ensheathed = extract_ensheathed_DAPI(DAPI_tmp, list_cells_sorted)
#    plt.imsave(s_path + 'DAPI_ensheathed_second' + str(im_num) + '.tif', (DAPI_ensheathed * 255).astype(np.uint16))
#    with open(s_path + 'DAPI_ensheathed' + str(im_num) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#       pickle.dump([DAPI_ensheathed], f)    
#       
#       
#       
#    """ Create for ==> GLOBAL JACCARD """
#    # first find all unique values
#    uniq = np.unique(new_fibers)
#    
#    binary_all_fibers = all_fibers > 0
#    labelled = measure.label(binary_all_fibers)
#    cc_overlap = measure.regionprops(labelled, intensity_image=all_fibers)
#    
#    final_counted = np.zeros(all_fibers.shape)
#    for Q in range(len(cc_overlap)):
#        cell_num = cc_overlap[Q]['MinIntensity']
#        cell_num = int(cell_num) 
#        overlap_coords = cc_overlap[Q]['coords']
#            
#        fiber = 0
#        for T in range(len(uniq)):
#            if cell_num == uniq[T]:
#                fiber = 1
#                #print(uniq[T])
#                break
#    
#        if fiber:
#            for T in range(len(overlap_coords)):
#                final_counted[overlap_coords[T,0], overlap_coords[T,1]] = cell_num
#
#    import pickle
#    with open(s_path + 'final_jacc_fibers' + str(im_num) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#       pickle.dump([final_counted], f)       
  

#make later to replace fiber_to_list: def multi_measure(input_arr, list_cells, ...
 
# loop through list_cells making at least one dataframe entry for each
    # if cell has no sheaths, leave NAs or whatever
    # if yes loop through each sheath, make entry for each one
    # also increase number of rows, such that dataframe will end up being
        # number of all cells + number of all sheaths (minus 1)
# save all
        
        
    # input the list of cells, unload attributes into a dataframe
        # dataframe has one entry per sheath plus blank entries for cells without sheaths
        # next fxn converts this into a more useable cell-based DF
        # setting it up like this to make it relatively easy to go btwn sheath and cells - 
            # should be able to (for ex) restrict cells easily by min sheath length
      
def perSheath_output_df(list_cells):

    cell = Cell(0)
    headings = [a for a in vars(cell)] # get list of cell attributes to use as headings
    headings.remove('coords')   # only used for overlap removal, no need now
    
    df = pd.DataFrame(columns=headings) # make the DF
    
    row = 0     # row counter - increase when adding fiber rows
    for i in range(len(list_cells)):
        cell = list_cells[i]
        df.loc[row,'num'] = cell.num
        
        #  df.loc[row,'intB'] = cell.intB[j]
        #  df.loc[row,'varB'] = cell.varB[j]
        #  df.loc[row,'minB'] = cell.minB[j]
        #  df.loc[row,'maxB'] = cell.maxB[j]
        
        #  df.loc[row,'centB'] = cell.centB[j]
        #  df.loc[row,'areaB'] = cell.areaB[j]
        #  df.loc[row,'circB'] = cell.circB[j]   

        if len(cell.fibers) > 0:   # if cell has fibers then make a new row for each
            for j in range(len(cell.fibers)):
                df.loc[row,'num'] = cell.num
                df.loc[row,'fibers'] = cell.fibers[j]
                df.loc[row,'xCent'] = cell.xCent[j]
                df.loc[row,'yCent'] = cell.yCent[j]
                df.loc[row,'intS'] = cell.intS[j]
                df.loc[row,'varS'] = cell.varS[j]
                df.loc[row,'minS'] = cell.minS[j]
                df.loc[row,'maxS'] = cell.maxS[j]
                
              #  df.loc[row,'intE'] = cell.intE[j]
              #  df.loc[row,'varE'] = cell.varE[j]
              #  df.loc[row,'minE'] = cell.minE[j]
              #  df.loc[row,'maxE'] = cell.maxE[j]
     
                row = row+1
        else: 
            df.loc[row,'fibers'] = 0
            row = row+1
        dfSheath = df.copy()
        
    return dfSheath

        
from scipy.spatial import ConvexHull
import numpy as np

def perCell_output_df(dfSheath):
    df = []
    ySumThresh = 3  # max number of pixels 2 sheath's x values can be apart to be summed in ySum measures
                    # this should be an arg in the function in the future
    
    headings = ['meanSInt','normSInt','varSInt','meanLength','maxLength','nSheaths','convexCent','convexTips','feretX','feretY']
    df = pd.DataFrame(columns=headings)
    
    nCells = dfSheath.num.max()
    for i in range(nCells):
        # get all rows where num = i
        # if length  = 1 then ignore
        sheaths = dfSheath[dfSheath['num'].isin({i})] # get all rows where num = the cell 
        # add soma measures
        if len(sheaths) == 1 and np.sum(sheaths.fibers) == 0:
            meanSint = np.NaN
            normSint = np.NaN
            varSint = np.NaN
            meanLength = np.NaN
            maxLength = np.NaN
            nSheaths = np.NaN
            centHull = np.NaN
            tipHull = np.NaN
            xRange = np.NaN
            yRange = np.NaN
            # make blank entry
            #print(i+ " is blank")
        else:   # need to validate that this works for single-sheath cells
            meanSint = sheaths.intS.mean()          #intensity variables
            normSarray = (sheaths.intS - sheaths.minS)/(sheaths.maxS - sheaths.minS)
            normSint = normSarray.mean()
            varSint = sheaths.varS.mean()
            
            meanLength = sheaths.fibers.mean()
            maxLength = sheaths.fibers.max()
            nSheaths = sheaths.fibers.count()
            # some sort of interpolated mode and/or median length??
            
            # y-sum... (mean, max, n, mode/median) - iteratively search all sheaths.xCent for any matches
                # make new array of summed matches, and new array of original sheaths minus summed matches
                # need a tolerance term...
            
            if len(sheaths.fibers) > 2:
                cents = np.stack([sheaths.xCent,sheaths.yCent],axis=1)   # reconstruct array of centroid points
                centHull = ConvexHull(cents).volume # note - script made for 3D, so .volume gives area, .area gives perimiter (chiaaante)
                xRange = (sheaths.xCent.max()-sheaths.xCent.min())
                
                tipsTop = np.stack((cents[:,0],(cents[:,1]-(sheaths.fibers/2)))) # estimate top and bottom coords of sheaths from centroids and lengths
                tipsBot = np.stack((cents[:,0],(cents[:,1]+(sheaths.fibers/2))))
                tips = np.concatenate((tipsTop,tipsBot),axis=1).transpose()  # verbose method to just recombine it all into a list of sheath tip points
                tipHull = ConvexHull(tips).volume
                yRange = (tipsTop.min()-tipsBot.max())
            else:
                centHull = np.NaN
                tipHull = np.NaN
                xRange = np.NaN
                yRange = np.NaN

        df.loc[i,'meanSInt'] = meanSint
        df.loc[i,'normSInt'] = normSint
        df.loc[i,'varSInt'] = varSint
        df.loc[i,'meanLength'] = meanLength
        df.loc[i,'maxLength'] = maxLength
        df.loc[i,'nSheaths'] = nSheaths
        df.loc[i,'convexCent'] = centHull
        df.loc[i,'convexTips'] = tipHull
        df.loc[i,'feretX'] = xRange
        df.loc[i,'feretY'] = yRange
            
    dfCell = df.copy()
        
    return dfCell


import glob
import pandas as pd


def sheathCsv_to_plateCellDf(dirInput, dirOutput):

    dirInput = "C:\\Users\\darya\\OneDrive - McGill University\\UNet\\190429M_UNet-01\\190519U_moreMeasures_analysis\\plate1_inputs\\"

    fileUnblind = glob.glob(dirInput+ "*unblinder.csv")
    dfUnblind = pd.read_csv(fileUnblind[0]) # glob makes list of files, so need to call first element even though there's only 1 element
    
    dfPlate = []
    
    fileList = glob.glob(dirInput+"sheath*.csv")
    if len(dfUnblind) != len(fileList): print("Warning - uneven # of csv's and unblinder rows")

    for i in range(len(fileList)):
        dfSheath = pd.read_csv(fileList[i]).drop(labels="Unnamed: 0",axis=1)    # import dataframe (drop first column of 0's)
        dfCell = perCell_output_df(dfSheath)
        
        indexUnblind = pd.DataFrame([dfUnblind.iloc[0].transpose()]*len(dfCell))
        dfCell.index = pd.MultiIndex.from_frame(indexUnblind)
        
        
        print("well " +str(i+1)+ " of " +str(len(fileList)))
        
        if (i == 0):   # if this is the first well, create a new big pooled dataframe now
            dfPlate = dfCell.copy()
        else:                   # if its not the first well and the big dataframe is made, just concatenate this well
            dfPlate = pd.concat([dfPlate,dfCell])
    
    
    # save big dfPlate
        
        
        #dfCell.set_index('num',inplace=True)    # sets cell # to be row index - can use this to align unblinder

    
    
    
    
    
    
    
    
    