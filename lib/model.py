# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 08:46:19 2024 
@author: cj_khoh
"""
  
import pandas as pd 
import numpy as np
 
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import LocalOutlierFactor 
from sklearn.neighbors import NearestNeighbors 
from sklearn.svm import OneClassSVM
from statsmodels.tsa.seasonal import STL 

class MachineLearning:
    def isolationForest(self, data): 
        if_model = IsolationForest()
        return np.where(if_model.fit_predict(data) == -1, 1, 0) 
            
    def dbscan(self, data, metric=np.std):
        # Choose MinPts (typically 4 or 5)
        MinPts = 4

        # Compute k-nearest neighbor distances
        nbrs = NearestNeighbors(n_neighbors=MinPts).fit(data)
        distances, _ = nbrs.kneighbors(data)

        # Sort distances (take the k-th column)
        sorted_distances = np.sort(distances[:, MinPts-1])
        
        eps_optimal = metric(sorted_distances)
        eps_optimal = max(eps_optimal, 0.1)

        dbscan = DBSCAN(eps=eps_optimal)          
 
        return np.where(dbscan.fit_predict(data) == -1, 1, 0)

    def local_outlier_factor(self, data):
        lof = LocalOutlierFactor()
        return np.where(lof.fit_predict(data) == -1, 1, 0)

    def hdbscan(self, data):
         # Choose MinPts (typically 4 or 5)
        MinPts = 5

        # Compute k-nearest neighbor distances
        nbrs = NearestNeighbors(n_neighbors=MinPts).fit(data)
        distances, _ = nbrs.kneighbors(data)

        # Sort distances (take the k-th column)
        sorted_distances = np.sort(distances[:, MinPts-1])
        
        eps_optimal = np.median(sorted_distances)
        
        hdbscan = HDBSCAN(cluster_selection_epsilon=eps_optimal)
        return np.where(hdbscan.fit_predict(data) == -1, -1, 1)
     
    def oneClassSVM(self, data):
        svm = OneClassSVM()
        return svm.fit_predict(data) == 1
 

def detect_point_anomalies(data: pd.Series, threshold: int, window_size: int) -> np.ndarray:
    mean_12_loop = data.rolling(window=window_size * 12).mean()
    mean_24_loop = data.rolling(window=window_size * 24).mean()
    
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


def detect_contextual_anomalies(data, metric=np.std):
    def isolationForest(): 
        if_model = IsolationForest()
        return np.where(if_model.fit_predict(data) == -1, 1, 0)
            
    def dbscan():
        eps_optimal = float(np.mean(metric(data, axis=0)))
        eps_optimal = max(eps_optimal, 0.1)
        
        dbscan = DBSCAN(eps=eps_optimal, min_samples=4)
        return np.where(dbscan.fit_predict(data) == -1, 1, 0)

    def local_outlier_factor():
        lof = LocalOutlierFactor()
        return np.where(lof.fit_predict(data) == -1, 1, 0)
    
    return isolationForest()
    

def detect_collective_anomalies():
    pass

