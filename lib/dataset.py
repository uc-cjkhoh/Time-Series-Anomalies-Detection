# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:26:20 2024 
@author: cj_khoh
"""
 
import sys
import pandas as pd  
import numpy as np 
from impala.dbapi import connect 
  
# Impala Server Configuration
IMPALA_HOST = 'VHEKPGNN-VIP'
IMPALA_PORT = 21050

class Dataset:
    def __init__(self, query:str = None):
        """
        Extract data from Apache Impala

        Args:
            query (STRING): SQL Query For Apache Impala . Defaults to None.
        """        
        
        if query == None:
            raise IOError("The SQL file is empty ...")
        
        self.query = query   
        
        try: 
            # try to connect
            conn = connect(host=IMPALA_HOST, port=IMPALA_PORT)
            cursor = conn.cursor()    
            cursor.execute(query)
            
            # try to get data
            self.data = pd.DataFrame(
                cursor.fetchall(), 
                columns=pd.DataFrame(cursor.description).iloc[:, 0].values
            ).sort_values('dt')
        except Exception as e:
            print(e)    
        finally:
            print("Data extracted successfully from Impala ...")
            
     
    def get_data(self, mcc_mnc:str, bound_type:int, rat_type:int) -> pd.DataFrame:  
        """
        Filter dataset with MCC, MNC, Bound Type, RAT Type

        Args:
            mcc_mnc (STRING): Country and Operator Code
            bound_type (INT): Inbound or Outbound
            rat_type (INT): Radio Access Technology Type

        Returns:
            DataFrame: subset of specific MCC, MNC, Bound Type, RAT Type
        """        
        
        subset = self.data[
            (self.data['mcc_mnc'] == mcc_mnc) & 
            (self.data['par_bound_type'] == bound_type) & 
            (self.data['rat_type'] == rat_type)
        ] 
        subset.index = np.arange(0, subset.shape[0], 1)
        subset.index.name = 'idx'
        
        return subset
        
    
    def get_mcc_list(self) -> np.ndarray:
        """
        Get all unique MCC, MNC Code

        Returns:
            Stirng: all unique country and corresponding operator code
        """        
        return np.sort(self.data['mcc_mnc'].unique())

        
    def get_bound_list(self) -> np.ndarray:
        """
        Get all bound type

        Returns:
            np.ndarray: all unique inbound or outbound code
        """        
        return np.sort(self.data['par_bound_type'].unique())

    
    def get_rat_list(self) -> np.ndarray:
        """
        Get all rat type

        Returns:
            np.ndarray: all unique rat type 
        """        
        return np.sort(self.data['rat_type'].unique())