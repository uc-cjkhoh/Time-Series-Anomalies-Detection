import sys
import pandas as pd  
import numpy as np 

from impala.dbapi import connect 
from prefect import task, get_run_logger
from prefect.cache_policies import NO_CACHE
  

class Impala:
    def __init__(self, host:str, port:int):
        """
        Initial Impala Connection
        
        Args:
            host (str): impala host number
            port (int): impala port number
        """
        
        self.host = host
        self.port = port
        self.conn = connect(host=self.host, port=self.port)
        self.cur = self.conn.cursor()
        
        
    @task(name='Retrieve Data', cache_policy=NO_CACHE)
    def get_records(self, query:str):
        """
        Extract data from Apache Impala

        Args:
            query (str): SQL Query For Apache Impala . Defaults to None.
        """        

        logger = get_run_logger()
        
        try: 
            self.cur.execute(query)
            self.data = pd.DataFrame(self.cur.fetchall())
            return self.data
            
        except Exception as err:
            logger.error(f'Error occurs: {err}', exc_info=True)
            raise
            
     
    @task(name='Retrieve Subdata', cache_policy=NO_CACHE)
    def get_subdata(self, mcc_mnc:str, bound_type:int, rat_type:int):  
        """
        Filter dataset with MCC, MNC, Bound Type, RAT Type

        Args:
            mcc_mnc (str): Country and Operator Code
            bound_type (int): Inbound or Outbound
            rat_type (int): Radio Access Technology Type
        """        
        
        subset = self.data[
            (self.data['mcc_mnc'] == mcc_mnc) & 
            (self.data['par_bound_type'] == bound_type) & 
            (self.data['rat_type'] == rat_type)
        ] 
        subset.index = np.arange(0, subset.shape[0], 1)
        subset.index.name = 'idx'
        
        return subset