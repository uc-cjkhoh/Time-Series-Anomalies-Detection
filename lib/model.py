# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 08:46:19 2024 
@author: cj_khoh
"""
  
import pandas as pd 
import numpy as np
 
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN 


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
    mean_12_loop = data.rolling(window=window_size * 12, min_periods=1).mean()
    mean_24_loop = data.rolling(window=window_size * 24, min_periods=1).mean()
      
    means = [mean_12_loop, mean_24_loop]
    std = data.std()
    
    final_result = np.zeros(data.shape[0])
    
    # calculate lower and upper boundaries
    for mean in means:
        lower_boundary = mean - (threshold * std)
        upper_boundary = mean + (threshold * std)

        # check if the data points are within the boundaries
        temp_result = ~((lower_boundary <= data) & (data <= upper_boundary))
        temp_result = np.where(temp_result, 1, 0)

        final_result += temp_result
    
    return final_result


def detect_contextual_anomalies(data, metric=np.mean):
    """
    Detect contextual anomalies in the data using various models.

    Args:
        data (_type_): The input data to be analyzed
        metric (_type_, optional): Metric for clustering model working with distancing. Defaults to np.std.

    Returns:
        np.ndarray: An array indicating the presence of anomalies (1 for anomaly, 0 for normal)
    """ 
    # isolation forest
    if_model = IsolationForest()
    isolation_forest_result =  pd.Series(np.where(if_model.fit_predict(data) == -1, 1, 0))
         
    return isolation_forest_result 
    
 