# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:26:20 2024 
@author: cj_khoh
"""
 
import sys
import pandas as pd  
import numpy as np 
from impala.dbapi import connect 
  
# configure and queries
IMPALA_HOST = 'VHEKPGNN-VIP'
IMPALA_PORT = 21050

class Dataset:
    def __init__(self, query=None):
        """_summary_

        Args:
            query (String, optional): Complete query to execute
        """
        
        self.query = query   
        
        # initialize connection
        conn = connect(host=IMPALA_HOST, port=IMPALA_PORT)
        cursor = conn.cursor()    
        cursor.execute(query)
        
        self.data = pd.DataFrame(
            cursor.fetchall(), 
            columns=pd.DataFrame(cursor.description).iloc[:, 0].values
        ).sort_values('dt')
        
        
        # self.data.to_csv('./anomaly_detection.csv')
     
    # return dataset
    def get_data(self, mcc, bound_type, rat_type):  
        subset = self.data[
            (self.data['mcc_mnc'] == mcc) & 
            (self.data['par_bound_type'] == bound_type) & 
            (self.data['rat_type'] == rat_type)
        ] 
        subset.index = np.arange(0, subset.shape[0], 1)
        subset.index.name = 'idx'
        
        return subset
        
    
    def get_mcc_list(self):
        return np.sort(self.data['mcc_mnc'].unique())

        
    def get_bound_list(self):
        return np.sort(self.data['par_bound_type'].unique())

    
    def get_rat_list(self):
        return np.sort(self.data['rat_type'].unique())

    def get_raw(self):
        return self.data
