import numpy as np
import pandas as pd

from datetime import datetime
from scipy.signal import detrend
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import STL


class FeatureEngineer:
    def __init__(self, data: pd.Series):
        self.data = data
    
    
    def mavg(self, nlags: int):
        """Perform N-lagged Moving Average"""
        return self.data.rolling(window=nlags, min_periods=1).mean()
    
    
    def autocorrelation(self, *args, **kwargs):
        """Perform Autocorrelation"""
        return acf(self.data, **kwargs)
        
        
    def partial_autocorrelation(self, *args, **kwargs):
        """Perform Partial-Autocorrelation"""
        return pacf(self.data, **kwargs)
        
    
    def fourier_transforms(self, sampling_interval: int):
        """Perform Fourier Transform"""
        detrend_data = detrend(self.data)
        
        fft_vals = np.fft.fft(detrend_data)
        freqs = np.fft.fftfreq(len(self.data), d=sampling_interval)
        
        is_pos = freqs > 0
        power = np.abs(fft_vals[is_pos]) ** 2
        
        peak_power = np.max(power)
        avg_power = np.mean(power)
        
        power_ratio = peak_power / avg_power
        
        return power_ratio
        
        # has_cycle = power_ratio > power_ratio_threshold
        
        # if has_cycle:
        #     peak_freq = freqs[np.argmax(power)]
        #     period_minutes = 1 / peak_freq
        #     return True, period_minutes, power_ratio
        # else:
        #     return False, None, power_ratio
            
    
    def decomposition(self, period: int):
        """Perform Seasonal Decomposition"""
        decomposed_data = STL(self.data, period=period, robust=True).fit()
        trend = decomposed_data.trend
        seasonal = decomposed_data.seasonal
        residual = decomposed_data.resid
        
        return trend, seasonal, residual