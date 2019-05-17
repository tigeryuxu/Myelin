# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:11:38 2018
@author: Neuroimmunology Unit
"""

#from sys import path
from os import getcwd
cur_dir = getcwd()
#path.append(cur_dir + "\\Data_functions") #Yes, i'm on windows

#from Data_functions import *
import Data_functions.test_network_UNet as UNet
import Data_functions.post_process_functions as post_process
from GUI import *

import logging
import traceback

import tkinter
from tkinter import filedialog
import os
    
debug = 0

""" LOAD UP GUI """
root = Tk()
my_gui = GUI(root)
root.mainloop()

im_scale = my_gui.scale
min_microns = my_gui.minLength
sensitivity = my_gui.sensitivity
rolling_ball = my_gui.rolling_ball
CLAHE = my_gui.CLAHE
resize = my_gui.resize

# if nothing entered, switch to default
if im_scale == None or min_microns == None or sensitivity == None or rolling_ball == None or resize == None:
    im_scale = '0.69'
    min_microns = '12'
    sensitivity = '3'
    rolling_ball = '0'
    CLAHE = '0'
    resize = '0'
    print("Nothing entered, switching to default")    
print("Parameters saved: " + "\nScale: " + im_scale + " \nminLength: " + min_microns + "\nSensitivity: " + sensitivity + 
      "\nRolling ball size: " + rolling_ball + "\nCLAHE: " + CLAHE + "\nresize: " + resize)

im_scale = float(im_scale)
min_microns = float(min_microns)
sensitivity = float(sensitivity)
rolling_ball = float(rolling_ball)
CLAHE = float(CLAHE)
resize = float(resize)

#min_microns = 12
#im_scale = 0.6904  #0.519, 0.6904, 0.35
minLength = min_microns / im_scale
minSingle = (minLength * sensitivity) / im_scale
minLengthDuring = 4/im_scale
#radius = 1.5/im_scale  # um   ==> can switch to 2 um (any lower causes error in dilation)
radius = 1.5/im_scale  # um   ==> can switch to 2 um (any lower causes error in dilation)


len_x = 1024     # 1344, 1024
width_x = 640   # 864, 640


channels = 3
green = 0


rand_rot = 0
rotate = 0         #*** 1024, 1024 for rotation
if rotate:
    width_x = 1024
jacc_test = 0



#s_path = filedialog.askdirectory(parent=root, initialdir=cur_dir,
#                                        title='Please select checkpoint directory')
#s_path = s_path + '/'

""" Prompt user to select input and output directories """
#""" Best so far is 980000 for rotated """
try:
    checkpoint = '401000'
    root = tkinter.Tk()
    s_path = './Checkpoints/'
    
    s_path = './Checkpoints/New-normalized-1024x1024/'; checkpoint = '594000'; width_x = 1024;
    #s_path = './Checkpoints/New-non-norm-1024x1024/'; checkpoint = '570000'; width_x = 1024; #must also comment out line 216!!!
    
    
    
    
    sav_dir = filedialog.askdirectory(parent=root, initialdir=cur_dir,
                                            title='Please select saving directory')
    sav_dir = sav_dir + '/'
    
    # get input folders
    another_folder = 'y';
    list_folder = []
    input_path = cur_dir
    while(another_folder == 'y'):
        input_path = filedialog.askdirectory(parent=root, initialdir= input_path,
                                            title='Please select input directory')
        input_path = input_path + '/'
        
    #    another_folder = input();   # currently hangs forever
        another_folder = 'n';
    
        list_folder.append(input_path)
except:
    print("Error in selecting input directories. Please re-load script.")


# make a saving directory for each input folder that is looped through
for i in range(len(list_folder)):

    name_folder = list_folder[i].split('/')
        
    sav_dir_folder = sav_dir + name_folder[-2] + '_output/'
    
    if not os.path.exists(sav_dir_folder):
        os.makedirs(sav_dir_folder)
    
    input_path = list_folder[i]
    
    try:
        UNet.run_analysis(s_path, sav_dir_folder, input_path, checkpoint,
                     im_scale, minLength, minSingle, minLengthDuring, radius,
                     len_x, width_x, channels, CLAHE, rotate, jacc_test, rand_rot, rolling_ball, resize,
                     debug)

        print("Analysis of image " + str(i + 1) + " successfully completed.")
        post_process.read_and_comb_csv_as_SINGLES(sav_dir_folder)  

    except Exception as error:
        print("Analysis of image " + str(i + 1) + " failed. Exiting program.")
        #print(error)
        logging.error(traceback.format_exc())
        
    # combines all individual output csv files into single excel sheets under folder "combined CSVs" in the output folder
    post_process.read_and_comb_csv_as_SINGLES(sav_dir_folder)
        
    

    
    
