import numpy as np
import pandas as pd
import model

from scipy import ndimage  
from statsmodels.tsa.seasonal import STL  
from scipy.stats import median_abs_deviation, percentileofscore

  
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
    Optimized for adaptive windowing, edge handling, and index alignment.

    Args:
        anomaly_result (pd.Series): The original result from anomaly detection (binary or integer, 0/1/2).
        window_size (int): The local window size (should be odd and >= 3). 

    Returns:
        pd.Series: Filtered anomaly indicator (same index as input, 2 for kept anomaly, 0 otherwise).
    """
    anomaly_result = pd.Series(anomaly_result)
    n = len(anomaly_result)
    
    # Ensure window_size is odd and not greater than n
    window = min(window_size, n)
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1 if window + 1 <= n else -1

    # Forward and backward rolling mean for density
    forward_local = anomaly_result.rolling(window=window, min_periods=1, center=True).mean()
    backward_local = anomaly_result[::-1].rolling(window=window, min_periods=1, center=True).mean()[::-1]
    
    # Take the maximum density from both directions
    final_score = np.maximum(forward_local, backward_local)
    
    # Only keep anomalies with density above threshold
    filtered = anomaly_result.astype(bool) & (final_score > threshold)
    
    # Return as Series with same index, 2 for anomaly, 0 otherwise
    return pd.Series(np.where(filtered, 1, 0), index=anomaly_result.index)


def get_quantiles_series(reference_data, target_value):
    vectorized_percentile = np.vectorize(
        lambda x: percentileofscore(reference_data, x) / 100
    )
    
    return vectorized_percentile(target_value)