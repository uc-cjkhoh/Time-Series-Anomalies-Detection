import pandas as pd 
import numpy as np
import util
   
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import STL 
from scipy.stats import median_abs_deviation, skew
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

 
def get_all_anomalies(residual, seasonal, trend, lower_q=0.01, upper_q=0.99): 
    mad = median_abs_deviation(residual, nan_policy='omit')
    Modified_Z = (residual - residual.median()) / (mad * 1.4826)
    
    # adaptive quantile-based threshold
    lower = Modified_Z.quantile(lower_q)
    upper = Modified_Z.quantile(upper_q)
     
    indicator_1 = pd.Series( 
        np.where(
            (Modified_Z < lower) | (Modified_Z > upper), 1, 0
        )
    ) 
    
    return indicator_1
    
     
def threshold_based_detector(data: pd.Series, window_size: int, threshold: float = 1.96, lower_q: float = 0.05, upper_q: float = 0.95) -> bool:    
    trend = STL(data, period=window_size // 24, robust=True).fit().trend
         
    diff = data - trend
    
    # what shoule be the optimal minimal std ? 
    local_std = data.rolling(window=window_size, min_periods=1).std().bfill().ffill().clip(lower=0.5 * np.std(diff))
    
    lower_boundary = trend - (threshold * local_std)
    upper_boundary = trend + (threshold * local_std)

    # check if the data points are within the boundaries
    indicator_1 = ~((lower_boundary <= data) & (data <= upper_boundary))
   
    # ===============================================================
    
    lower_quantile = pd.Series(diff).rolling(window=window_size, min_periods=1).apply(
        lambda x : np.quantile(x, lower_q)
    ).bfill().ffill()
    
    indicator_2 = (diff > lower_quantile)
   
    return indicator_1 & indicator_2 


def mad_based_z_score(data, window_size, lower_q=0.025, upper_q=0.975) -> bool : 
    polynomial_fit = STL(data, period=window_size, robust=True).fit().trend 
    
    # Calculate deviation from trend
    deviation = data - polynomial_fit
    
    # create a MAD-based Z-score on the deviations)
    mad = median_abs_deviation(deviation)
    # mad = pd.Series(deviation).rolling(window_size, min_periods=1).apply(
    #     lambda x: median_abs_deviation(x, nan_policy='omit')
    # ).bfill().ffill()
     
    deviation_median = deviation.median()
    # deviation_median = pd.Series(deviation).rolling(window_size, min_periods=1).median()
     
    Modified_Z = (deviation - deviation_median) / (mad * 1.4826)
    
    # adaptive quantile-based threshold 
    lower = Modified_Z.quantile(lower_q) 
    upper = Modified_Z.quantile(upper_q)
    
    return (Modified_Z < lower) | (Modified_Z > upper), Modified_Z


def get_contextual(data, all_anomalies, window_size, lower_q=0.025, upper_q=0.975): 
    # Use STL decomposition for better seasonal pattern recognition 
    decompose = STL(data, period=window_size, robust=True).fit()
    expected_value = decompose.trend + decompose.seasonal
     
    smoothed_data = STL(data, period=window_size // 24, robust=True).fit().trend   
    
    # Calculate deviation from trend
    deviation = smoothed_data - expected_value
      
    mad = median_abs_deviation(deviation)
    # mad = pd.Series(deviation).rolling(window_size, min_periods=1).apply(
    #     lambda x: median_abs_deviation(x, nan_policy='omit')
    # ).bfill().ffill()
     
    deviation_median = deviation.median()
    # deviation_median = pd.Series(deviation).rolling(window_size, min_periods=1).median()
     
    Modified_Z = (deviation - deviation_median) / (mad * 1.4826) 
    
    # adaptive quantile-based threshold
    lower = Modified_Z.quantile(lower_q)
    upper = Modified_Z.quantile(upper_q)
    
    # determine if a data point is not in IQR
    out_of_mad = (Modified_Z < lower) | (Modified_Z > upper)
    
    contextual = np.where(all_anomalies.astype(bool) & (out_of_mad), 1, 0)
    all_anomalies += contextual 
     
    # ======================================== 
    
    # anomalies_idx = np.where(all_anomalies == 1)[0]
    # idx_diff = np.abs(pd.Series(anomalies_idx).diff().fillna(0))
    
    # # seasonal_strength = util.check_seasonality(data, period=window_size)
    # # trend_strength = util.check_trend(data, period=window_size)     
    # # strength = max(seasonal_strength, trend_strength)
         
    # eps = max(1, idx_diff.sum() / len(anomalies_idx))
    # min_samples = (len(all_anomalies[all_anomalies == 1]) // 24) // eps
     
    # print("EPS: ", eps)
    # print("Min Samples: ", min_samples)
    
    # db = DBSCAN(
    #     eps=eps,
    #     min_samples=min_samples
    # ).fit_predict(anomalies_idx.reshape(-1, 1))
    
    # db_result = np.where(db == -1, 1, 0)
    
    # all_anomalies[anomalies_idx] += db_result
    
    return pd.Series(all_anomalies), smoothed_data, expected_value, Modified_Z