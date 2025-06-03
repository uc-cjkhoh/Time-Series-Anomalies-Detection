# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 08:46:19 2024 
@author: cj_khoh
"""
  
import pandas as pd 
import numpy as np
 
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN 
from scipy.signal import savgol_filter
from scipy import ndimage


def detect_point_anomalies(data: pd.Series, threshold: int, window_size: int) -> np.ndarray: 
    """
    Detect point anomalies in the data using a rolling mean and standard deviation.

    Args:
        data (pd.Series): The input data to be analyzed
        threshold (int): The threshold for anomaly detection
        window_size (int): The size of the rolling window

    Returns:
        np.ndarray: An array indicating the presence of anomalies (1 for anomaly, 0 for normal)
    """
    mean_12_loop = savgol_filter(data, window_length=window_size * 12, polyorder=2, mode='wrap')
    mean_24_loop = savgol_filter(data, window_length=window_size * 24, polyorder=2, mode='wrap')
    
    # mean_12_loop = data.rolling(window=window_size * 12, min_periods=1).mean()
    # mean_24_loop = data.rolling(window=window_size * 24, min_periods=1).mean()
      
    means = [mean_12_loop, mean_24_loop]
    std = data.std()
    
    final_result = np.zeros(data.shape[0])
    
    # calculate lower and upper boundaries
    for mean in means:
        lower_boundary = mean - (threshold * std)
        upper_boundary = mean + (threshold * std)

        # check if the data points are within the boundaries
        temp_result = ~((lower_boundary <= data) & (data <= upper_boundary))
        temp_result = np.where(temp_result, 1, 0)  # Convert boolean to 2 for anomalies

        final_result += temp_result
    
    return final_result


def detect_contextual_anomalies(data, threshold, window_size):
    """
    Detect contextual anomalies in the data using various models.

    Args:
        data (_type_): The input data to be analyzed
        metric (_type_, optional): Metric for clustering model working with distancing. Defaults to np.std.

    Returns:
        np.ndarray: An array indicating the presence of anomalies (1 for anomaly, 0 for normal)
    """ 
    
    smoothed_data = savgol_filter(data, window_length=window_size, polyorder=2, mode='wrap')
    
    # iso = IsolationForest()
    # iso_anomalies = iso.fit_predict(smoothed_data.reshape(-1, 1)) == -1
    # iso_anomalies = np.where(iso_anomalies, 1, 0)  # Convert boolean to 2 for anomalies
    
    median = ndimage.median(data)
    constant_std = data.std()
    moving_std = data.rolling(window=12, min_periods=1).std() 
    
    # calculate lower and upper boundaries
    lower_boundary = median - (threshold * constant_std)
    upper_boundary = median + (threshold * constant_std)

    # check if the data points are within the boundaries
    constant_result = ~((lower_boundary <= smoothed_data) & (smoothed_data <= upper_boundary))
    constant_result = np.where(constant_result, 1, 0)  # Convert boolean to 2 for anomalies

    lower_boundary = median - (threshold * moving_std)
    upper_boundary = median + (threshold * moving_std)

    moving_result = ~((lower_boundary <= smoothed_data) & (smoothed_data <= upper_boundary))
    moving_result = np.where(constant_result, 1, 0)  # Convert boolean to 2 for anomalies
 
    return constant_result, moving_result