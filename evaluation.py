# -*- coding: utf-8 -*-
"""
python script

Calculate the estimation accuracy of each measurement point.
    1. 50 points with estimated values around the measurement points were sampled. 
    2. caluculat the average value of the sampling points.

@author: kmatsui
"""
## import library
import numpy as np


## define function
def get_average(wq, site):
    """
    Caluculat the average value (mean) of the sampling points.

    Parameters
    ----------
    wq : ndarray
        Water qualiy values.
    site : array
        Data that stores coordinate of 50 points

    Returns
    -------
    ndarray
        Mean of water quality values sampled from 50 point.

    """
    val = []
    for coor in site:
        tmp = coor.split()
        val.append(wq[int(tmp[2]), int(tmp[1])])

    val = np.array(val)
    return np.mean(val[np.nonzero(val)])


def main(pred, mask, coor):
    """
    Calculate the estimation accuracy of each measurement point.
        1. 50 points with estimated values around the measurement points were sampled. 
        2. caluculat the average value of the sampling points.
    
    Measurement sites:
        - site 1, ogata bridge
        - site 2, east adjustment pond
        - site 3, center of the adjustment pond
        - site 4, west adjustment pond
        - site 5, floodgate        

    Parameters
    ----------
    data : ndarray
        Prediction result by neural network.
    mask : ndarrya
        Mask data.
    coor : array
        Coorinate of measurement site in Lake hachiroko.
        The coordinates are (0,0) at the upper left of the image data.
        Used for evaluation

    Returns
    -------
    res_average : dataFrame
        Average values (mean) of estimated water quality value at
        five measurement sites.

    """    
    res_average = [0]*5 # array to store mean values
    
    ## convert to image format
    mask = mask//255    
    hh, ww = mask.shape
    pred = pred.reshape((hh, ww))    
    pred = pred*mask
    
    ## 50 points with estimated values around the measurement points were sampled. 
    ## Coordinates were randomly determined in advance.
    data_st1 = coor[ 1: 51]  # ogata bridge
    data_st2 = coor[ 52:102] # east adjustment pond
    data_st3 = coor[205:255] # center of the adjustment pond
    data_st4 = coor[103:153] # west adjustment pond
    data_st5 = coor[154:204] # floodgate        
    
    ## caluculat the average value of the sampling points.
    res_average[0] = get_average(pred, data_st1) 
    res_average[1] = get_average(pred, data_st2) 
    res_average[2] = get_average(pred, data_st3) 
    res_average[3] = get_average(pred, data_st4) 
    res_average[4] = get_average(pred, data_st5) 
    
    return res_average
