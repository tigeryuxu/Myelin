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

import zipfile

from skimage.morphology import skeletonize
from skimage.morphology import *
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import mahotas as mah
import numpy as np
import cv2

from plot_functions import *
from data_functions import *
from post_process_functions import *
from UNet import *


min_microns = 12
im_scale = 0.6904  #0.519, 0.6904, 0.35
minLength = min_microns / im_scale
minSingle = (minLength) / im_scale    # otherwise * 3 if want to set lower sensitivity threshold
minLengthDuring = 4/im_scale
radius = 3/im_scale  # um


""" defines a cell object for saving output """
class Cell:
    def __init__(self, num):
        self.num = num
        self.fibers = []    # creates a new empty list for each dog

    def add_fiber(self, fibers):
        self.fibers.append(fibers)
        

person = 'Mat_jacc'
skel_bool = 0
input_path = 'D:/Tiger/AI stuff/MyelinUNet/Testing/Valid_fibers/'
#DAPI_path='C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Testing/DAPI/'

# Read in file names
onlyfiles_mask = [ f for f in listdir(input_path) if isfile(join(input_path,f))]   
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
onlyfiles_mask.sort(key = natsort_key1)
counter = list(range(len(onlyfiles_mask)))  # create a counter, so can randomize it

#for i in range(len(onlyfiles_mask)):   
input_arr = readIm_counter(input_path,onlyfiles_mask, counter[0]) 

truth_im = np.asarray(input_arr)

binary = truth_im > 0
#skeleton = binary

""" SKEL or no SKEL??? """
if skel_bool:
    skeleton = skeletonize(binary)
else:
    skeleton = binary

""" Initiate list of CELL OBJECTS """
N = 5000;
num_MBP_pos = N

list_cells_v = []
for i in range(N):
    cell = Cell(N)
    list_cells_v.append(cell)

""" Eliminate anything smaller than minLength, and in wrong orientation, then add to cell object """
#minLength = 18
#binary_all_fibers = truth_im > 0
labelled = measure.label(skeleton)
cc_overlap = measure.regionprops(labelled, intensity_image=truth_im)

final_counted_truth = np.zeros(truth_im.shape)
for i in range(len(cc_overlap)):
    length = cc_overlap[i]['MajorAxisLength']
    angle = cc_overlap[i]['Orientation']
    overlap_coords = cc_overlap[i]['coords']

    #print(angle)
    if length > minLength and (angle > +0.785398 or angle < -0.785398):
        #print(angle)
        cell_num = cc_overlap[i]['MinIntensity']
        cell_num = int(cell_num) 
        
        list_cells_v[cell_num].add_fiber(length)

        for T in range(len(overlap_coords)):
            final_counted_truth[overlap_coords[T,0], overlap_coords[T,1]] = cell_num

cmap = plt.cm.jet
#plt.imsave('truth_5_' + person + '.tif', (final_counted_truth * 255).astype(np.uint16))

with open('truth_5_final_fibers' + person + '-' + str(minLength) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
     pickle.dump([final_counted_truth], f)     

""" go through list_cells to get all the information """
#minLengthSingle = 0;
output_name = 'truth_5_' + person + '-' + str(minLength) + '.csv'
#cycle_and_output_csv(list_cells_v, output_name, minLengthSingle)

cycle_and_output_csv(list_cells_v, output_name, minSingle, total_DAPI=0, total_matched_DAPI=0, s_path='')
