# -*- coding: utf-8 -*-
"""
python script

Create a water quality map from the prediction results by neural network.
The water quality Map is colored by 6 colors and black (mask area).

@author: kmatsui
"""
## import library
import numpy as np
from progressbar import progressbar as pbar

## define function
def convert_wq2degree(pred, levels):
    """
    Convert a water quality value to the pollution degree

    Parameters
    ----------
    pred : ndarray
        Predict values of water quality.
    levels : array
        slice levels of water quality.

    Returns
    -------
    degree : ndarray
        Pllution degree (level 0~6).

    """
    max_lv = levels[len(levels)-1]
    re_levels = np.array(levels) / max_lv
    re_pred = pred / max_lv
    degree = np.zeros((len(re_pred), 1))

    for ii in pbar(range(len(re_pred))):
        if re_pred[ii, 0] < re_levels[0]: # level 1
            degree[ii, 0] = 1
        elif re_levels[0] <= re_pred[ii, 0] and re_pred[ii, 0] < re_levels[1]: # level 2
            degree[ii, 0] = 2
        elif re_levels[1] <= re_pred[ii, 0] and re_pred[ii, 0] < re_levels[2]: # level 3
            degree[ii, 0] = 3
        elif re_levels[2] <= re_pred[ii, 0] and re_pred[ii, 0] < re_levels[3]: # level 4
            degree[ii, 0] = 4
        elif re_levels[3] <= re_pred[ii, 0] and re_pred[ii, 0] < re_levels[4]: # level 5
            degree[ii, 0] = 5
        elif re_levels[4] <= re_pred[ii, 0]: # level 6
            degree[ii, 0] = 6
        else:
            degree[ii, 0] = 0
    
    return degree
            

def main(data, mask):
    """
    Create a water quality map from the prediction results by neural network
    The water quality Map is colored by 6 colors and black (mask area).

    Parameters
    ----------
    data : ndarray
        Prediction result by neural network.
    mask : ndarrya
        Mask data.

    Returns
    -------
    img : ndarray
        Water quality map.

    """
    
    
    """
    Variable definition:
        - WQ_levels:
            slice levels of water quality. In this program, the slice levels
            of SS is used.
        - color:
            colors corresponding to the pollution degree.
            level0 (mask region, lowest pollution), black
            level1, blue
            level2, green
            level3, yellow
            level4, orange
            level5, red
            level6 (highest pollution), white        
        - hh, ww: height and width of image data for analysis
    
    """  
    WQ_levels = [1, 5, 15, 25, 35] # water quality (SS) levels
    color = [[  0,  0,  0], # level 0
             [255,  0,  0], # level 1
             [  0,230,  0], # level 2
             [  0,255,255], # level 3
             [  0,125,255], # level 4
             [  0,  0,255], # level 5
             [255,255,255]] # level 6
    hh, ww = mask.shape
    
    
    """
    Create the water quality map:
        1. convert a water quality value to the pollution degree
        2. convert to image format
    
    """  
    ## convert a water quality value to the pollution degree
    cl = convert_wq2degree(data, WQ_levels)
    
    cl = cl.reshape((hh, ww))
    cl = cl * mask//255
    cl = cl.astype('uint8')
    
    ## convert to image format
    color = np.array(color)
    img = np.zeros((hh, ww, 3))    
    for ii, hh in enumerate(cl):
        for jj, ww in enumerate(hh):
            img[ii, jj] = color[ww]
    
    img = np.array(img.astype('uint8'))
    
    return img

