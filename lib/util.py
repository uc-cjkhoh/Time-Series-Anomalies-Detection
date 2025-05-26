# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:38:15 2024 
@author: cj_khoh
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import acf 
from statsmodels.tsa.seasonal import STL  

def replace_point_anomalies(data: pd.Series, point_anomalies_labels: pd.Series, window_size: int = 5) -> pd.DataFrame:
    """
    Replace the confirm point anomalies by Moving Average

    Args:
        data (pd.DataFrame): the target data
        point_anomalies_labels (np.array): the result from point anomalies detection
        window_size (int): the window size to operate moving average

    Returns:
        pd.DataFrame: return the dataset without point anomalies
    """
    rolling_mean = data.rolling(window_size, min_periods=1).mean()
    data = np.where(point_anomalies_labels.values != 0, data, rolling_mean)
    
    return data

   
# contextual anomaly: scale data & smooth data 
def smoothing(data: pd.Series, window_size: int = 5) -> pd.Series:
    """
    Smooth the data using a moving average filter.
    
    Args:
        data (pd.DataFrame): The input data to be smoothed
        window_size (int, optional): The size of the moving window. Defaults to 5.

    Returns:
        pd.DataFrame: The smoothed data
    """ 
     
    scaler = MinMaxScaler()
      
    data = pd.Series(scaler.fit_transform(data.values.reshape(-1, 1)).flatten())
    data = data.rolling(window=window_size, min_periods=1).mean()
    
    return data 


# check if the data has seasonality
def check_seasonality(data: pd.Series, period: int) -> bool:
    """
    Check if the data has seasonality using STL decomposition.
    
    Args:
        data (pd.DataFrame): The input data to be checked
        period (int, optional): The period for seasonal decomposition.

    Returns:
        bool: True if the data has seasonality, False otherwise
    """ 
    # check if a series of data has seasonality, 1 for yes, 0 for no
    # no_of_cycle = 5
    
    # acf_result = acf(data, nlags=period*no_of_cycle)
    
    # has_seasonality = True if acf_result[np.arange(period, period*no_of_cycle, period)].mean() >= 0.5 else False
      
    stl = STL(data, period=period).fit()
    has_seasonality = True if max(0, 1 - (np.var(stl.resid) / np.var(stl.resid + stl.seasonal))) > 0.5 else False
      
    return has_seasonality 
