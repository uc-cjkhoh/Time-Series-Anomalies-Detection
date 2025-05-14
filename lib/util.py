# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:38:15 2024 
@author: cj_khoh
"""

import os 
from datetime import datetime
 
# 2. log the current process 
def logging(message, filepath='anomaly_log.txt'):
    try:
        with open(filepath, 'a') as log_file:
            log_file.write(message)
            log_file.write('\n')
            
        log_file.close()
            
    except Exception as e:
        print(f"Error: {e}")    
         