import numpy as np
import pandas as pd

from scipy import ndimage  
from statsmodels.tsa.seasonal import STL  


def replace_point_anomalies(target_data: pd.Series, point_result: pd.Series, replacement_data: pd.Series) -> pd.Series:
    """
    Replace point anomalies in the target data with values from the replacement data.
    
    Args:
        target_data (pd.Series): The data in which anomalies are to be replaced
        replacement_data (pd.Series): The data to replace anomalies with

    Returns:
        pd.Series: The target data with anomalies replaced
    """ 
    # Ensure both series are of the same length
    if len(target_data) != len(replacement_data):
        raise ValueError("Target and replacement data must have the same length.")
    
    replaced_column = np.where(point_result == 1, replacement_data, target_data)
    
    return pd.Series(replaced_column, index=target_data.index)
  

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
    stl = STL(data, period=period).fit()
    
    return max(0, 1 - (np.var(stl.resid) / np.var(stl.resid + stl.seasonal))) 
    # print(1 - (np.var(stl.resid) / np.var(stl.resid + stl.seasonal)))
    

# check if the data is strongly trend or not
def check_trend(data: pd.Series, period: int) -> bool:
    """
    Check if the data is strongly trend using STL decomposition.
    
    Args:
        data (pd.DataFrame): The input data to be checked
        period (int, optional): The period for trend decomposition.

    Returns:
        bool: True if the data is strongly trend, False otherwise
    """  
    stl = STL(data, period=period).fit()
    
    return max(0, 1 - (np.var(stl.resid) / np.var(stl.resid + stl.trend)))  
    

# calculate the density of anomalies
def anomaly_density(anomaly_result, window_size, threshold):
    """
    Calculate the density of anomalies and filter them based on a threshold.

    Args:
        anomaly_result (pd.Series): the orginal result from anomaly detection
        window_size (int): the local window size
        threshold (float): the number of stndard deviation away from the median

    Returns:
        _type_: _description_
    """
    
    forward_local = anomaly_result.rolling(window_size, center=True, min_periods=1).mean()  
    backward_local = anomaly_result[::-1].rolling(window_size, center=True, min_periods=1).mean()
    
    final_score = np.maximum(forward_local, backward_local[::-1])
   
    return np.where(anomaly_result.astype(bool) & (final_score > threshold), 1, 0)