import pandas as pd 
import numpy as np
import util
   
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import STL 


def detect_extreme_value(data: pd.Series, threshold: int, window_size: int) -> np.ndarray: 
    """
    Detect point anomalies in the data using a rolling mean and standard deviation.

    Args:
        data (pd.Series): The input data to be analyzed
        threshold (int): The threshold for anomaly detection
        window_size (int): The size of the rolling window

    Returns:
        np.ndarray: An array indicating the presence of anomalies (1 for anomaly, 0 for normal)
    """ 
    # mean_12_loop = savgol_filter(data, window_length=window_size * 12, polyorder=2, mode='mirror')
    trend = savgol_filter(data, window_length=window_size, polyorder=2, mode='mirror')
     
    # stl_decompose = STL(data, period=window_size, robust=True).fit()
    
    # trend = stl_decompose.trend 
    
    local_std = data.rolling(window=window_size, min_periods=1).std().clip(lower=data.std())
    
    # local_mean = pd.Series(trend).rolling(window=window_size * 24, min_periods=1).mean()
     
    lower_boundary = trend - (threshold * local_std)
    upper_boundary = trend + (threshold * local_std)

    # check if the data points are within the boundaries
    result = ~((lower_boundary <= data) & (data <= upper_boundary))
    result = np.where(result, 2, 0)  # Convert boolean to 2 for anomalies
   
    return pd.Series(result)


def detect_residual_outlier(data, threshold, window_size, trend_strength, seasonality_strength):
    """
    Detect contextual anomalies in the data using various models.
    Formula: LOESS(data) ± (sigma * median std of (residual + trend / seasonal [if strength >= 0.4]))

    Args:
        data (pd.Series): The input data to be analyzed
        threshold (float): The number of standard deviation away from mean

    Returns:
        np.ndarray: An array indicating the presence of anomalies (1 for anomaly, 0 for normal)
    """ 
     
    stl_decompose = STL(data, period=window_size, seasonal_deg=0, trend_deg=0, low_pass_deg=0, robust=True).fit() 
    
    target_data = data
    x = np.zeros(data.shape[0])
    y = pd.Series(stl_decompose.resid)

    if trend_strength >= 0.4:
        x += stl_decompose.trend
        y += stl_decompose.trend
    else:
        target_data -= stl_decompose.trend
    
    if seasonality_strength >= 0.4:
        x += stl_decompose.seasonal
        y += stl_decompose.seasonal
    else:
        target_data -= stl_decompose.seasonal
    
    if trend_strength < 0.4 and seasonality_strength < 0.4:
        return x
    
    y = y.rolling(window=window_size, min_periods=1).std()
        
    lower_bound = x - threshold * y
    upper_bound = x + threshold * y
     
    # median = ndimage.median(data)
    # constant_std = data.std()  
        
    # check if the data points are within the boundaries
    result = ~((lower_bound <= target_data) & (target_data <= upper_bound))
    final_result = np.where(result, 1, 0)  # Convert boolean to 2 for anomalies
    
    
    return pd.Series(final_result) 
