import pandas as pd

from datetime import datetime
from scipy.fft import fft, rfft, fftfreq, rfftfreq
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import STL


class FeatureEngineer:
    def __init__(self, data: pd.Series):
        self.data = data
    
    
    def mavg(self, nlags: int):
        """Perform N-lagged Moving Average"""
        return self.data.rolling(window=nlags, min_periods=1).mean()
    
    
    def autocorrelation(self, nlags: int):
        """Perform Autocorrelation"""
        return acf(self.data, nlags=nlags)
        
        
    def partial_autocorrelation(self, nlags: int):
        """Perform Partial-Autocorrelation"""
        
    
    def fourier_transforms(self, sampling_interval: int):
        """Perform Fourier Transform"""
        return fft(self.data)
    
    
    def decomposition(self, period: int):
        """Perform Seasonal Decomposition"""
        decomposed_data = STL(self.data, period=period, robust=True).fit()
        trend = decomposed_data.trend
        seasonal = decomposed_data.seasonal
        residual = decomposed_data.resid
        
        return trend, seasonal, residual
    
    
    def extract_features(self): 
        return {
            'mavg_1': self.mavg(1),
            'mavg_7': self.mavg(7),
            'mavg_14': self.mavg(14),
            'mavg_28': self.mavg(28),

        }