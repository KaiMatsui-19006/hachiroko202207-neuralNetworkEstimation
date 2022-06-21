# -*- coding: utf-8 -*-
"""
python script

Learning process by neural network.
    1. data normalization
    2. neural network learning process
    3. save loss plot and model
    4. act model to test data 

@author: kmatsui
"""
## import library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## tensorflow library
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

## limit memory utilization
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")


## define function
def meanStd(data):
    """
    Normalize the train data for each of remote sensing data, water depth, 
    and water temperature based on the mean and standard deviation.

    Parameters
    ----------
    data : ndarray
        Data to be normalized, train data.

    Returns
    -------
    re_data : ndarray
        Data after normalization.
    mean : float
        Mean used for normalization.
    std : float
        Standard deviation used for normalization.

    """
    mean = np.mean(data)
    std = np.std(data)

    re_data = (data - mean)/std

    return re_data, mean, std

def meanStd_test(data, mean, std):
    """
    Test data is normalized based on the mean and standard deviation
    of the train data.

    Parameters
    ----------
    data : ndarray
        Data to be normalized, test data.
    mean : float
        Mean used for normalization.
    std : float
        Standard deviation used for normalization.

    Returns
    -------
    re_data : ndarray
        Data after normalization.

    """
    re_data = (data - mean)/std

    return re_data

def nn_learning(input_size, output_size, hidden_size, activation, out_activation, droprate):
    """
    Construct a model of neural network. Model is constructed based on arguments.
    Other parameter:
        - optimizer: Adam
        - loss function: mean squared error

    Parameters
    ----------
    input_size : int
        Number of units in the input layer.
    output_size : int
        Number of units in the output layer.
    hidden_size : int
        Number of units in the hidden layer.
    activation : string
        Activate function to the hidden layer.
    out_activation : string
        Activate function to the output layer.
    droprate : float
        Dropout rate.

    Returns
    -------
    model : tensorflow
        Network model.

    """

    model = Sequential([
                        Dense(hidden_size, input_dim=input_size, activation=activation),
                        Dropout(droprate),
                        Dense(hidden_size, activation=activation),
                        Dropout(droprate),
                        Dense(hidden_size, activation=activation),
                        Dropout(droprate),
                        Dense(output_size, activation=out_activation)
                        ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model


def main(x_train, y_train, x_test):
    """
    Learning process by neural network.
        1. data normalization
        2. neural network learning process
        3. save loss plot and model
        4. act model to test data
    
    Parameters
    ----------
    x_train : ndarry
        train input dataset
    y_train : ndarray
        train output dataset
    x_test : ndarray
        test input dataset


    Returns
    -------
    predictions : ndarray
        estimated water quality values.

    """
    
    
    """
    Data normalization:
        Normalize the train data for each of remote sensing data, water depth, 
        and water temperature based on the mean and standard deviation.
        Test data is also normalized based on the mean and standard deviation
        of the train data.
        
    """        
    ## train input data
    x_train[:, 0:9], meanDN, stdDN = meanStd(x_train[:, 0:9])
    x_train[:, 9:18], meanDepth, stdDepth = meanStd(x_train[:, 9:18])
    x_train[:, 18:27], meanTemperature, stdTemperature = meanStd(x_train[:, 18:27])
    
    ## test input data
    x_test[:, 0:9] = meanStd_test(x_test[:, 0:9], meanDN, stdDN)
    x_test[:, 9:18] = meanStd_test(x_test[:, 9:18], meanDepth, stdDepth)
    x_test[:, 18:27] = meanStd_test(x_test[:, 18:27], meanTemperature, stdTemperature)
    
    
    """
    Neural network learning process:
        model configurations:
            - Number of hidden layers: 3 
            - Number of units in hidden layers: 900  
            - Activation function:
                - hidden layer: ReLU
                - output layer: linear
            - Dropout layer: 0.5
            
        other parameters:
            - batch size: 256
            - epochs: 1000
            - validation data: 10% of train data
        
        Save the chackpoint to "./dst/nn_results".
        
    """        
    model = nn_learning(x_train.shape[1], 1,
                        hidden_size=900,
                        activation="relu",
                        out_activation="linear",
                        droprate=0.5)
    model.summary()
    
    ## make checkpoint
    checkpoint_dpath = "./dst/nn_results/modelCheck"
    
    model_chekpoint = ModelCheckpoint(
        filepath=checkpoint_dpath,
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=True,
        period=1)
    
    ## start learning process
    history = model.fit(x_train, y_train,
                        batch_size=256, epochs=1_000,
                        validation_split=0.1,
                        shuffle=True,
                        callbacks=[model_chekpoint])
    
    
    """
    Save loss plot and model:
        Save the loss plot and created model to "./dst/nn_results".
        
    """       
    hist_df = pd.DataFrame(history.history)
    
    ## loss plot
    plt.figure()
    hist_df[['loss', 'val_loss']].plot()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig("./dst/nn_results/loss_plot.png")
    plt.close()
    
    ## model
    json_string = model.to_json()
    open('./dst/nn_results/model.json', 'w').write(json_string)
    
    
    """
    Act model to test data:
        Apply the created model to the test data.
        Since the sample data is SS of a water quality parameter, multiply 
        the prediction result by 35 (pre-set max value) and convert it to 
        water quality values.
        
    """
    predictions = model.predict(x_test)
    predictions = predictions*35
    
    return predictions
