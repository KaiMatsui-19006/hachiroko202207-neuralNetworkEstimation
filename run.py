# -*- coding: utf-8 -*-
"""
python script

Please run this program.
NN results, water quality map, and evaluation result are created 
in the "./dst" directory.

@author: kmatsui
"""
## import library
import pickle
import cv2
import os
import pandas as pd

## import local fucntion
import nn_learning
import create_map
import evaluation


## define function
def load_pickle(path):
    """
    load data from local directory
    
    Parameters
    ----------
    path : string
        Target data path.

    Returns
    -------
    data : 
        Target data.

    """
    with open(path,'rb') as f:
        data=pickle.load(f)
    return data

def save_pickle(data, path):
    """
    save data to local directory
    
    Parameters
    ----------
    data : 
        Target data.
    path : string
        Data path.

    Returns
    -------
    None.

    """
    with open(path,'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    """
    This program call and execute the programs of learning process,
    creating water quality map, and evaluation.
    """

    """
    x data: ndarray (number of sample, input size (27))
        Input data is 3x3 pixel: 
            remote sensing data [:, 0:9], water depth [:, 9:18], water temperature [:, 18:27]  
            
    y data: ndarray (number of sample)
        Output data is 1 pixel:
            water quality value
    
    mask data: ndarray (height, width: 668, 689)
        Used for cration of water quality map and evaluation.
        
    coordinate: array
        Coorinate of measurement site in Lake hachiroko.
        The coordinates are (0,0) at the upper left of the image data.
        Used for evaluation
    
    """    
    ## input
    x_train = load_pickle("./sample/train_input.pkl")
    x_test = load_pickle("./sample/test_input.pkl")    
    
    ## output
    y_train = load_pickle("./sample/train_output.pkl")
    
    ## mask
    mask = load_pickle("./sample/mask.pkl")
    
    ## coordinate
    f = open('./sample/evalation_points.txt', 'r')    
    coordinate = f.readlines()    
    f.close()
    
    os.makedirs("./dst", exist_ok=True)
    
    """
    Learning process by neural network:
        Save the nn results to "./dst/nn_results".
    
    """   
    pred = nn_learning.main(x_train, y_train, x_test)
    save_pickle(pred, "./dst/nn_results/pred_results.pkl")
    
    
    """
    Create the water quality map:
        Save the water quality map (.png) to "./dst".
    
    """   
    img = create_map.main(pred, mask)
    cv2.imwrite("./dst/water_quality_map.png", img)


    """
    Evaluation the water quality map:
        Save the evaluation resutls (.xlsx) to "./dst".
    
    """   
    evalate = evaluation.main(pred, mask, coordinate)
    df = pd.DataFrame(evalate,
                      index=["site1", "site2", "site3", "site4", "site5"],
                      columns=["proposed method"])
    
    with pd.ExcelWriter('./dst/evaluate_proposed.xlsx') as writer:
        df.to_excel(writer, sheet_name='mean')