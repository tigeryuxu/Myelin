# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:35:56 2019

@author: darya

- as of now contains functions for the following image operations:
    - rescaling
    - gaussian blur
    - noise
    - contrast 
    - stretch in X, Y dimensions
- tried to set it up so each has 2 parameters - first chooses the type (ie 
    Gaussian noise vs salt and pepper), second is the magnitude (generally btwn 0
    and 1)
    - only 1 type of blur for now (probably don't need more than this..) so only 1 
    parameter there for the sigma
    - scale2 can be >1 if you want to upscale

- other to-do's
    - make real functions 
    - reduce dependencies
    - add in frequency parameter for all operations, to choose how many images to 
    randomly apply them to (ie if scale3 = 0.2, 20% of images get rescaled)
"""


import numpy as np
import skimage as ski

import skimage.io
import scipy 
import matplotlib.pyplot as plt

from random import randint, choice
from math import floor

im = skimage.io.imread("C:/Users/darya/Desktop/JbbHeart_vf.tif")


# rescale, 0=none 1=constant 2=random
# second term is scale factor (can be >1), or random range of scaling (btwn 0-1)
# side note - did wierd multiplying-then-dividing-by-10 thing so that 
    # scale2 could be an integer for randint, then go back to being <1
scale1 = 2
scale2 = 0.5
import skimage.transform

if scale1 == 1:
    imScaled = skimage.transform.rescale(im,scale2,anti_aliasing=True,multichannel=True)
if scale1 ==2: 
    imScaled = skimage.transform.rescale(im,scale2/randint(1,floor(scale2*10))/10, anti_aliasing=True,multichannel=True)


# gaussian blur
# note - gaussian function has other optional parameters including wrap
blur = 2
import skimage.filters

imBlurred = skimage.filters.gaussian(im,sigma=blur,multichannel=True)


# gaussian noise, type is first term (0=none, 1=gaussian, 2=speckle, 3=s&p)
# ideally add other noise in future (speckle, s&p could be useful)
# note - seems like speckle in general is best at emulating immunofluorescence noise
noise1 = 3
noise2 = 0.5
import skimage.util

noise2 = noise2*0.1   # was WAY too strong so in interest of keeping things consistent
                      # this way if noise2=0.5, 5% of pixels will become noise instead
                      # of 50                     
if noise1 == 1:
    imNoised = skimage.util.random_noise(im,mode="gaussian",var=noise2)
if noise1 == 2:
    imNoised = skimage.util.random_noise(im,mode="speckle",var=noise2)
if noise1 == 3:
    imNoised = skimage.util.random_noise(im,mode="s&p",amount=noise2)


# contrast (0 = none, 1 = linear equalize, 2 = adaptive equalize, 3 = random)
# random magnitude is percentage of image range to cover, ie cont2 = 0.1 means
    # image can shift up by up by 5% and down by up to 5%; 50% means 25% on each end
# should rewrite random to increase OR decrease floor and ceiling, as was done with stretch below
# should also allow channel(s) to be specified; does all flattened channels now
cont1 = 3
cont2 = 0.5
import skimage.exposure

if cont1 == 1:
    imContrast = skimage.exposure.equalize_hist(im)
if cont1 == 2:
    imContrast = skimage.exposure.equalize_adapthist(im,clip_limit=cont2)
if cont1 == 3:
    imMin = im.min(axis=0).min()
    imMax = im.max(axis=0).max()
    imRange = imMax-imMin
    newMin = imMin + randint(0,floor(cont2*imRange/2))
    newMax = imMax - randint(0,floor(cont2*imRange/2))
    imContrast = skimage.exposure.rescale_intensity(im,in_range='image',out_range=(newMin.astype(int),newMax.astype(int)))

# stretch image (0 = none, 1 = X dimension only, 2 = Y only, 3 = both)
# second term is randomness magnitude (btwn 0 and 1) ie if stretch2 = 0.5, then
    # images will be stretched up by 50%
# may need to add in something to reshape image to original size after...
stretch1 = 3
stretch2 = 0.5
import skimage.transform

if stretch1 == 1:
    if choice([True, False]) == True:           # randomly decides if neg or pos stretching
        imStretch = skimage.transform.resize(im,(im.shape[0]*stretch2,im.shape[1]))
    else:
        imStretch = skimage.transform.resize(im,(im.shape[0]/stretch2,im.shape[1]))
        
if stretch1 == 2:
    if choice([True, False]) == True:         
        imStretch = skimage.transform.resize(im,(im.shape[1],im.shape[1]*stretch2))
    else:
        imStretch = skimage.transform.resize(im,(im.shape[1],im.shape[1]/stretch2))
        
if stretch1 == 3:
    if choice([True, False]) == True:   
        imStretch = skimage.transform.resize(im,(im.shape[0]*stretch2,im.shape[1]))
    else:
        imStretch = skimage.transform.resize(im,(im.shape[0]/stretch2,im.shape[1]))
        
    if choice([True, False]) == True:           
        imStretch = skimage.transform.resize(im,(im.shape[0],im.shape[1]*stretch2))
    else:
        imStretch = skimage.transform.resize(im,(im.shape[0],im.shape[1]/stretch2))
        
