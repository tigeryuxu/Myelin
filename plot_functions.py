# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:16:39 2017

@author: Tiger
"""


import tensorflow as tf
import math
import pylab as mpl
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import random
from skimage import measure

""" ADDS TEXT TO IMAGE and saves the image """
def add_text_to_image(all_fibers, filename='default.png', resolution=800):
    #fiber_img = Image.fromarray((all_fibers *255).astype(np.uint16)) # ORIGINAL, for 8GB CPU
    fiber_img = (all_fibers*255).astype(np.uint16) 
    plt.figure(80, figsize=(12,10)); plt.clf(); plt.imshow(fiber_img)
    plt.axis('off')
    # PRINT TEXT ONTO IMAGE
    binary_all_fibers = all_fibers > 0
    labelled = measure.label(binary_all_fibers)
    cc_overlap = measure.regionprops(labelled, intensity_image=all_fibers)
    
    # Make a list of random colors corresponding to all the cells
    list_fibers = []
    for Q in range(int(np.max(all_fibers) + 1)):
        color = [random.randint(0,255)/256, random.randint(0,255)/256, random.randint(0,255)/256]
        list_fibers.append(color)
        
    for Q in range(len(cc_overlap)):
        overlap_coords = cc_overlap[Q]['coords']
        new_num = cc_overlap[Q]['MinIntensity']
        
        #if cell_num != new_num:
            #color = [random.randint(0,255)/256, random.randint(0,255)/256, random.randint(0,255)/256]
            #cell_num = new_num
        color = list_fibers[int(new_num)]
        plt.text(overlap_coords[0][1], overlap_coords[0][0], str(int(new_num)), fontsize= 2, color=color)    
    plt.savefig(filename, dpi = resolution)

"""
    Scales the normalized images to be within [0, 1], thus allowing it to be displayed
"""
def show_norm(im):
    m,M = im.min(),im.max()
    plt.imshow((im - m) / (M - m))
    plt.show()


""" Originally from Intro_to_deep_learning workshop
"""

def plotOutput(layer,feed_dict,fieldShape=None,channel=None,figOffset=1,cmap=None):
	# Output summary
	W = layer
	wp = W.eval(feed_dict=feed_dict);
	if len(np.shape(wp)) < 4:		# Fully connected layer, has no shape
		temp = np.zeros(np.product(fieldShape)); temp[0:np.shape(wp.ravel())[0]] = wp.ravel()
		fields = np.reshape(temp,[1]+fieldShape)
	else:			# Convolutional layer already has shape
		wp = np.rollaxis(wp,3,0)
		features, channels, iy,ix = np.shape(wp)   # where "features" is the number of "filters"
		if channel is not None:
			fields = wp[:,channel,:,:]
		else:
			fields = np.reshape(wp,[features*channels,iy,ix])    # all to remove "channels" axis

	perRow = int(math.floor(math.sqrt(fields.shape[0])))
	perColumn = int(math.ceil(fields.shape[0]/float(perRow)))
	fields2 = np.vstack([fields,np.zeros([perRow*perColumn-fields.shape[0]] + list(fields.shape[1:]))])    # adds more zero filters...
	tiled = []
	for i in range(0,perColumn*perRow,perColumn):
		tiled.append(np.hstack(fields2[i:i+perColumn]))    # stacks horizontally together ALL the filters

	tiled = np.vstack(tiled)    # then stacks itself on itself
	if figOffset is not None:
		mpl.figure(figOffset); mpl.clf(); 

	mpl.imshow(tiled,cmap=cmap); mpl.title('%s Output' % layer.name); mpl.colorbar();
    
    
""" Plot layers
"""
def plotLayers(feed_dict, L1, L2, L3, L4, L5, L6, L8, L9, L10):
      plt.figure('Down_Layers');
      plt.clf()
      plt.subplot(221); plotOutput(L1,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(222); plotOutput(L2,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(233); plotOutput(L3,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(234); plotOutput(L5,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(223); plotOutput(L4,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.pause(0.05)
      
      plt.figure('Up_Layers');
      plt.clf()
      plt.subplot(221); plotOutput(L6,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(222); plotOutput(L8,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(223); plotOutput(L9,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(224); plotOutput(L10,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.pause(0.05); 
      
  
""" Plots global and detailed cost functions
""" 
  
def plot_cost_fun(plot_cost, plot_cost_val, plot_cost_val_NO=None):
      """ Graph global loss
      """      
      plt.figure(18); plt.clf();
      plt.plot(plot_cost, label='Training'); plt.title('Global Loss')
      plt.ylabel('Loss'); plt.xlabel('Epochs'); plt.pause(0.05)
      
      # cross-validation
      plt.figure(18); plt.plot(plot_cost_val, label='Cross_validation'); plt.pause(0.05)
      plt.legend(loc='upper left');    
      
      """ Graph detailed plot
      """
      last_loss = len(plot_cost)
      start = 0
      if last_loss < 50:
          start = 0
      elif last_loss < 200:
          start = last_loss - 50
          
      elif last_loss < 500:
          start = last_loss - 200
          
      elif last_loss < 1500:
          start = last_loss - 500
          
      else:
          start = last_loss - 1500
      plt.close(19);
      x_idx = list(range(start, last_loss))
      plt.figure(19); plt.plot(x_idx,plot_cost[start:last_loss], label='Training'); plt.title("Detailed Loss"); 
      plt.figure(19); plt.plot(x_idx,plot_cost_val[start:last_loss],label='Cross_validation');
      plt.legend(loc='upper left');             
      plt.ylabel('Loss'); plt.xlabel('Epochs'); plt.pause(0.05)    
      
      if plot_cost_val_NO is not None:
            plt.figure(18); plt.plot(plot_cost_val_NO, label='Cross_validation_NO'); plt.pause(0.05)                                      
            plt.figure(19); plt.plot(x_idx, plot_cost_val_NO[start:last_loss], label='Cross_validation_NO');   plt.pause(0.05)    
      
        
""" Plots global and detailed cost functions
""" 
  
def plot_jaccard_fun(plot_jaccard, plot_jaccard_val=False):
      """ Graph global jaccard
      """      
      plt.figure(21); plt.clf();
      plt.plot(plot_jaccard, label='Jaccard'); plt.title('Jaccard')  
      if plot_jaccard_val:
          plt.plot(plot_jaccard_val, label='Cross Validation Jaccard');
      plt.ylabel('Jaccard'); plt.xlabel('Epochs');            
      plt.legend(loc='upper left');    plt.pause(0.05)
      
      
      
def plot_overlay(plot_cost, plot_cost_val, plot_jaccard, plot_cost_val_NO=None):
      """ Graph global loss
      """      
      plt.figure(18); 
      
      #plt.clf();
      plt.plot(plot_cost, label='Training_NO_W'); plt.title('Global Loss')
      plt.ylabel('Loss'); plt.xlabel('Epochs'); plt.pause(0.05)
      
      # cross-validation
      plt.figure(18); plt.plot(plot_cost_val, label='Cross_validation_NO_W'); plt.pause(0.05)
      plt.legend(loc='upper left');    
      
      """ Graph detailed plot
      """
      last_loss = len(plot_cost)
      start = 0
      if last_loss < 50:
          start = 0
      elif last_loss < 200:
          start = last_loss - 50
          
      elif last_loss < 500:
          start = last_loss - 200
          
      elif last_loss < 1500:
          start = last_loss - 500
          
      else:
          start = last_loss - 1500
      
      #plt.close(19);
      x_idx = list(range(start, last_loss))
      plt.figure(19); plt.plot(x_idx,plot_cost[start:last_loss], label='Training_NO_W'); plt.title("Detailed Loss"); 
      plt.figure(19); plt.plot(x_idx,plot_cost_val[start:last_loss],label='Cross_validation_NO_W');
      plt.legend(loc='upper left');             
      plt.ylabel('Loss'); plt.xlabel('Epochs'); plt.pause(0.05)    
      
      if plot_cost_val_NO is not None:
            plt.figure(18); plt.plot(plot_cost_val_NO, label='Cross_validation_NO'); plt.pause(0.05)                                      
            plt.figure(19); plt.plot(x_idx, plot_cost_val_NO[start:last_loss], label='Cross_validation_NO');   plt.pause(0.05)    
      
    
      plt.figure(21); 
      
      #plt.clf();
      plt.plot(plot_jaccard, label='Jaccard_NO_W'); plt.title('Jaccard')
      plt.ylabel('Jaccard'); plt.xlabel('Epochs'); 
      plt.legend(loc='upper left');    plt.pause(0.05)
  
    
    
""" Plots the moving average that is much smoother than the overall curve"""
    
def calc_moving_avg(plot_data, num_pts = 20, dist_points=100):
    
    new_plot = []
    for T in range(0, len(plot_data)):
        
        avg_points = []
        for i in range(-dist_points, dist_points):
            
            if T + i < 0:
                continue;
            elif T + i >= len(plot_data):
                break;
            else:
                avg_points.append(plot_data[T+i])
                
        mean_val = sum(avg_points)/len(avg_points)
        new_plot.append(mean_val)
        
    return new_plot
            
        
    
def change_scale_plot():

    multiply = 10
    font_size = 11
    legend_size = 11
    plt.rcParams.update({'font.size': 9})    
    """Getting back the objects"""
    plot_cost = load_pkl(s_path, 'loss_global.pkl')
    plot_cost_val = load_pkl(s_path, 'loss_global_val.pkl')
    plot_jaccard = load_pkl(s_path, 'jaccard.pkl')


    x_idx = list(range(0, len(plot_cost) * multiply, multiply));   
    plt.figure(19); plt.plot(x_idx,plot_cost, label='Training_weighted'); 
    #plt.title("Detailed Loss"); 
    plt.figure(19); plt.plot(x_idx,plot_cost_val,label='Validation_weighted');
    plt.legend(loc='upper right');             
    plt.ylabel('Loss', fontsize = font_size); plt.xlabel('Epochs', fontsize = font_size); plt.pause(0.05) 
    
    x_idx = list(range(0, len(plot_jaccard) * multiply, multiply));   
    plt.figure(20); plt.plot(x_idx,plot_jaccard, label='Validation_weighted'); 
    #plt.title("Detailed Loss");     
    plt.ylabel('Jaccard', fontsize = font_size); plt.xlabel('Epochs', fontsize = font_size); plt.pause(0.05) 
    plt.legend(loc='upper left');             
    
    """Getting back the objects"""
    plot_cost_noW = load_pkl(s_path, 'loss_global_no_W.pkl')
    plot_cost_val_noW = load_pkl(s_path, 'loss_global_val_no_W.pkl')
    plot_jaccard_noW = load_pkl(s_path, 'jaccard_no_W.pkl')
    

    x_idx = list(range(0, len(plot_cost_noW) * multiply, multiply));   
    plt.figure(19); plt.plot(x_idx,plot_cost_noW, label='Training_no_weight'); 
    #plt.title("Loss"); 
    plt.figure(19); plt.plot(x_idx,plot_cost_val_noW,label='Validation_no_weight');
    plt.legend(loc='upper right', prop={'size': legend_size});             

    
    x_idx = list(range(0, len(plot_jaccard_noW) * multiply, multiply));   
    plt.figure(20); plt.plot(x_idx,plot_jaccard_noW, label='Validation_no_weight'); 
    #plt.title("Jaccard");     
    plt.legend(loc='upper left', prop={'size': legend_size});      
    
    """ Calculate early stopping beyond 180,000 """
    plot_short = plot_cost_val[30000:-1]
    hist_loss = plot_short
    patience_cnt = 0    
    for epoch in range(len(plot_short)):
        # ... 
        # early stopping

        patience = 100
        min_delta = 0.02
        if epoch > 0 and hist_loss[epoch-1] - hist_loss[epoch] > min_delta:
            patience_cnt = 0
        else:
            patience_cnt += 1
     
        if patience_cnt > patience:
            print("early stopping...")
            print(epoch * 5 + 30000 * 5)
            break
    """ 204680 """
    
    """ MOVING AVERAGE """
    num_pts = 10
    dist_points = 200
    mov_cost = calc_moving_avg(plot_cost, num_pts=num_pts, dist_points=dist_points)
    mov_cost_val = calc_moving_avg(plot_cost_val, num_pts=num_pts, dist_points=dist_points)
    mov_jaccard = calc_moving_avg(plot_jaccard, num_pts=num_pts, dist_points=dist_points)

    
    font_size = 11
    plt.rcParams.update({'font.size': 10})    

    x_idx = list(range(0, len(mov_cost) * multiply, multiply));   
    plt.figure(21); plt.plot(x_idx,mov_cost, label='Training_weighted'); plt.title("Detailed Loss"); 
    plt.figure(21); plt.plot(x_idx,mov_cost_val,label='Validation_weighted');
    plt.legend(loc='upper left');             
    plt.ylabel('Loss', fontsize = font_size); plt.xlabel('Epochs', fontsize = font_size); plt.pause(0.05) 
    
    x_idx = list(range(0, len(mov_jaccard) * multiply, multiply));   
    plt.figure(22); plt.plot(x_idx,mov_jaccard, label='Validation_weighted'); plt.title("Detailed Loss");     
    plt.ylabel('Jaccard', fontsize = font_size); plt.xlabel('Epochs', fontsize = font_size); plt.pause(0.05) 
    plt.legend(loc='upper left');             
    
    """Getting back the objects"""
    num_pts = 10
    dist_points = 400
    mov_cost_noW = calc_moving_avg(plot_cost_noW, num_pts=num_pts, dist_points=dist_points)
    mov_cost_val_noW = calc_moving_avg(plot_cost_val_noW, num_pts=num_pts, dist_points=dist_points)
    mov_jaccard_noW = calc_moving_avg(plot_jaccard_noW, num_pts=num_pts, dist_points=dist_points)


    x_idx = list(range(0, len(mov_cost_noW) * multiply, multiply));   
    plt.figure(21); plt.plot(x_idx,mov_cost_noW, label='Training_no_weight'); plt.title("Loss"); 
    plt.figure(21); plt.plot(x_idx,mov_cost_val_noW,label='Validation_no_weight');
    plt.legend(loc='upper left');             

    
    x_idx = list(range(0, len(mov_jaccard_noW) * multiply, multiply));   
    plt.figure(22); plt.plot(x_idx,mov_jaccard_noW, label='Validation_no_weight'); plt.title("Jaccard");     
    plt.legend(loc='upper left');      
    
    

""" Plot the average for the NEWEST MyQz11 + ClassW + No_W"""

def change_scale_plot2():

    s_path = 'C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Checkpoints/ALL_FOR_PLOT/'
    
    multiply = 10
    font_size = 11
    legend_size = 11
    plt.rcParams.update({'font.size': 9})    
    
    """Getting back the objects"""
    #plot_cost = load_pkl(s_path, 'loss_global.pkl')
    plot_cost_val = load_pkl(s_path, 'loss_global_MyQz10_classW.pkl')
    plot_jaccard = load_pkl(s_path, 'jaccard_MyQz10_classW.pkl')

    """Getting back the objects"""
    #plot_cost_noW = load_pkl(s_path, 'loss_global_no_W.pkl')
    plot_cost_val_noW = load_pkl(s_path, 'loss_global_MyQ9_noW.pkl')
    plot_jaccard_noW = load_pkl(s_path, 'jaccard_MyQ9_noW.pkl')


    """Getting back the objects"""
    #plot_cost_noW = load_pkl(s_path, 'loss_global_no_W.pkl')
    plot_cost_val_sW = load_pkl(s_path, 'loss_global_MyQz11_sW_batch2.pkl')
    plot_jaccard_sW = load_pkl(s_path, 'jaccard_MyQz11_sW_batch2.pkl')


    font_size = 11
    plt.rcParams.update({'font.size': 10})  
    
    """ no-weight """
    dist_points_loss = 3
    dist_points_jacc = 25
    multiply = 1500
    #mov_cost_noW = calc_moving_avg(plot_cost_noW, num_pts=num_pts, dist_points=dist_points)
    mov_cost_val_noW = calc_moving_avg(plot_cost_val_noW, num_pts=num_pts, dist_points=dist_points_loss)
    mov_jaccard_noW = calc_moving_avg(plot_jaccard_noW, num_pts=num_pts, dist_points=dist_points_jacc)
       
    plot_single_cost(mov_cost_val_noW, multiply, 'Validation no weight', 'Loss')    
    plot_single_jacc(mov_jaccard_noW, multiply, 'Validation no weight', 'Jaccard')

    
    """ class weight """
    multiply = 1500
    #mov_cost = calc_moving_avg(plot_cost, num_pts=num_pts, dist_points=dist_points)
    mov_cost_val = calc_moving_avg(plot_cost_val, num_pts=num_pts, dist_points=dist_points_loss)
    mov_jaccard = calc_moving_avg(plot_jaccard, num_pts=num_pts, dist_points=dist_points_jacc) 
           
    plot_single_cost(mov_cost_val, multiply, 'Validation class weight', 'Loss')    
    plot_single_jacc(mov_jaccard, multiply, 'Validation class weight', 'Jaccard')
    

    """ spatial W """
    multiply = 1000
    #mov_cost_noW = calc_moving_avg(plot_cost_noW, num_pts=num_pts, dist_points=dist_points)
    mov_cost_val_noW = calc_moving_avg(plot_cost_val_sW, num_pts=num_pts, dist_points=dist_points_loss)
    mov_jaccard_noW = calc_moving_avg(plot_jaccard_sW, num_pts=num_pts, dist_points=dist_points_jacc)
       
    plot_single_cost(mov_cost_val_noW, multiply, 'Validation spatial weight', 'Loss')    
    plot_single_jacc(mov_jaccard_noW, multiply, 'Validation spatial weight', 'Jaccard')




def plot_single_cost(data, multiply, label, title):
    x_idx = list(range(0, len(data) * multiply, multiply));   
    plt.figure(21); plt.plot(x_idx,data, label=label); plt.title(title);     
    plt.legend(loc='upper left');     
    
def plot_single_jacc(data, multiply, label, title):
    x_idx = list(range(0, len(data) * multiply, multiply));   
    plt.figure(22); plt.plot(x_idx,data, label=label); plt.title(title);     
    plt.legend(loc='upper left');     
    