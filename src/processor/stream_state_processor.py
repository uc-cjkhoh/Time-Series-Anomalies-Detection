import numpy as np

from collections import OrderedDict
from river import stats, time_series, anomaly, preprocessing, linear_model, optim
from river.utils import Rolling


class StatsReport:
    def __init__(self, mcc, mnc, rat, bound_type, dt = None, tx_succ_count = 0, tx_total_count = 0):
        self.mcc = mcc
        self.mnc = mnc
        self.rat = rat
        self.bound_type = bound_type
        
        self.dt = dt
        self.tx_succ_count = tx_succ_count
        self.tx_total_count = tx_total_count
        
        
    def record_tx(self, tx_status):
        self.tx_total_count += 1
        self.tx_succ_count += tx_status
        
        
    def reset_tx(self, dt):
        self.dt = dt
        self.tx_succ_count = 0
        self.tx_total_count = 0
        

    def get_mcc(self):
        return self.mcc

    
    def get_mnc(self):
        return self.mnc

    
    def get_rat(self):
        return self.rat

    
    def get_bound_type(self):
        return self.bound_type


    def get_dt(self):
        return self.dt

    
    def get_tx_succ_count(self):
        return self.tx_succ_count
    
    
    def get_tx_total_count(self):
        return self.tx_total_count

    
    def get_report(self):
        return {
            'dt': self.dt,
            'mcc': self.mcc,
            'mnc': self.mnc,
            'rat_type': self.rat,
            'bound_type': self.bound_type,
            'succ_count': self.tx_succ_count,
            'total_count': self.tx_total_count
        }
        
    
# class LRUCache:
#     def __init__(self, capacity: int):
#         self.cache = OrderedDict()
#         self.capacity = capacity
        
    
#     def get(self, key: int):
#         if key not in self.cache:
#             return -1
#         else:
#             self.cache.move_to_end(key)
#             return self.cache[key]
        
    
#     def put(self, key: int, value: int):
#         self.cache[key] = value
#         self.cache.move_to_end(key)
#         if len(self.cache) > self.capacity: 
#             self.cache.popitem(last = False)