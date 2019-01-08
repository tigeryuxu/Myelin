# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:11:38 2018

@author: Neuroimmunology Unit
"""

from sys import path
from os import getcwd
path.append(getcwd() + "\\Data functions") #Yes, i'm on windows
print(path)
import test_network as UNet
from GUI import *

import tkinter
from tkinter import filedialog
import os
    
debug = 0


min_microns = 12
im_scale = 0.6904  #0.519, 0.6904, 0.35
minLength = min_microns / im_scale
minSingle = (minLength * 3) / im_scale
minLengthDuring = 4/im_scale
radius = 1.5/im_scale  # um   ==> can switch to 2 um (any lower causes error in dilation)

len_x = 1024     # 1344, 1024
width_x = 640   # 864, 640

channels = 3
CLAHE = 0
green = 0


rand_rot = 0
rotate = 0         #*** 1024, 1024 for rotation
if rotate:
    width_x = 1024
jacc_test = 0



""" LOAD UP GUI """
root = Tk()
my_gui = GUI(root)
root.mainloop()

im_scale = my_gui.scale
min_microns = my_gui.minLength
sensitvity = my_gui.sensitivity



""" Best so far is 980000 for rotated """
checkpoint = '301000'

root = tkinter.Tk()
s_path = filedialog.askdirectory(parent=root, initialdir="/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Checkpoints/",
                                        title='Please select checkpoint directory')
s_path = s_path + '/'

sav_dir = filedialog.askdirectory(parent=root, initialdir="/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Source/",
                                        title='Please select saving directory')
sav_dir = sav_dir + '/'

# get input folders
another_folder = 'y';
list_folder = []
input_path = "/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Source/"
while(another_folder == 'y'):
    input_path = filedialog.askdirectory(parent=root, initialdir= input_path,
                                        title='Please select input directory')
    input_path = input_path + '/'
    
#    another_folder = input();   # currently hangs forever
    another_folder = 'n';

    list_folder.append(input_path)



# make a saving directory for each input folder that is looped through
for i in range(len(list_folder)):

    name_folder = list_folder[i].split('/')
        
    sav_dir_folder = sav_dir + name_folder[-2] + '/'
    
    if not os.path.exists(sav_dir_folder):
        os.makedirs(sav_dir_folder)
    
    input_path = list_folder[i]
    

    UNet.run_analysis(s_path, sav_dir_folder, input_path, checkpoint,
                 im_scale, minLength, minSingle, minLengthDuring, radius,
                 len_x, width_x, channels, CLAHE, rotate, jacc_test, rand_rot,
                 debug)

        
    
    tf.reset_default_graph()
    
    
    
    
