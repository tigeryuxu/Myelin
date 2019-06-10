# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:02:04 2019

@author: darya
"""

import glob
import pandas as pd
from scipy.spatial import ConvexHull
import numpy as np


    """
    - input the dfSheath (entry for each sheath + empty for cells with no sheaths)
    - outputs new df where per-cell averages for many other parameters are each row
    - this will be nested in the below function which will call it for each well in a set

    - how to add measurements
        - add to "headings" array in beginning
        - create variable with value during loop through dfSheaths
        - print the value to dfCells during the next loop (df.loc)
        
    - cell parameters calculated:
        - intensity (S)
            - mean S int
            - norm S int (mean-min/max-min per sheath)
            - S var
        - lengths
            - max sheath length
            - mode and/or median sheath length (interpolated?)
            - mean sheath length
            - nsheaths
        - sheath distributions (for cells with 3 or more)
            - convex hull centroids
            - convex hull sheath tips
            - feret X 
            - feret Y
    - future
        - intensity (E [other channels]) - same as S
        - ratiometrics
            - S int / E int
            - S norm / E norm
            - covariance
        - soma measurements
        - intra-cell densities
            - avg sheath distance from soma
            - skew (soma centroid relative to total center of mass)
            - same but for convex area instead of center mass (area not intensity)
        - inter-cell densities
            - avg sheath distance from soma
            - avg soma distance from ensheathing somas
            - avg soma distance from O4 somas
            - avg soma distance from O4- somas
            - avg from all somas
            - avg distance from other sheaths
            - soma well coordinates
    - steps
        - make dataframe with headings
        - get # cells from num column
        - iterate through every row of dfSheaths - if fibers = 0 just log an empty cell (will still use soma measures in the future)
        
    """       

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



"""
- input a folder with sheathMeasurements csvs and unblinder (ie the save folder of UNet)
- uses "sheath" and "unblinder" to find each so make sure those are in th names
- imports each .csv and runs toCells_output_df
- unloads data to a new big dataframe, concatenating each well's cells to the last
- import unblinder .csv with columns like: cell #, treat, prep, duplicate, etc
- apply unblinder as multiindex (or new columns?)
- export monster csv

- to-dos
    - after adding to toCells fxn, add minLength and ySum args here to pass down

- notes
    - should only be 1 unblinder file, plus a "sheath" file for each well, in input folder
"""

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
    
    
    # save big dfPlate, multi-index
        
        
        #dfCell.set_index('num',inplace=True)    # sets cell # to be row index - can use this to align unblinder
