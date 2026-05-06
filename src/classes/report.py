import numpy as np 

from dataclasses import dataclass
from datetime import datetime
from river import anomaly, drift, stats
from river.utils import Rolling



class PeriodicReport:
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
         

@dataclass
class ReportSnapshot:
    mcc: int
    mnc: str
    rat: int
    bound_type: int
    dt: datetime
    tx_succ_count: int
    tx_total_count: int
     
    @property
    def report_id(self):
        return f'{self.mcc}-{self.mnc}-{self.rat}-{self.bound_type}'
    


class OnlineMLReport:
    def __init__(self, window_size):
        self.succ_tx_median = stats.RollingQuantile(q=0.5, window_size=window_size)
        self.succ_tx_var = Rolling(stats.Var(), window_size)
        self.succ_tx_autocorr = stats.AutoCorr(lag=window_size)
        
        self.total_tx_median = stats.RollingQuantile(q=0.5, window_size=window_size)
        self.total_tx_var = Rolling(stats.Var(), window_size)
        self.total_tx_autocorr = stats.AutoCorr(lag=window_size)
    
    
    def update_and_extract(self, snapshot: ReportSnapshot):
        current_succ_count = snapshot.tx_succ_count
        current_total_count = snapshot.tx_total_count 
        
        # success transaction count statistic
        self.succ_tx_median.update(current_succ_count) # type: ignore
        self.succ_tx_var.update(current_succ_count)
        self.succ_tx_autocorr.update(current_succ_count) # type: ignore
        
        succ_tx_median = self.succ_tx_median.get() 
        succ_tx_var = self.succ_tx_var.get() 
        succ_tx_autocorr = self.succ_tx_autocorr.get() 
        
        # total transcation count statistic
        self.total_tx_median.update(current_total_count) # type: ignore
        self.total_tx_var.update(current_total_count)
        self.total_tx_autocorr.update(current_total_count) # type: ignore
        
        total_tx_median = self.total_tx_median.get() 
        total_tx_var = self.total_tx_var.get() 
        total_tx_autocorr = self.total_tx_autocorr.get()
        
        return {
            'succ_tx_median': succ_tx_median,
            'succ_tx_var': succ_tx_var,
            'succ_tx_autocorr': succ_tx_autocorr,
            'total_tx_median': total_tx_median,
            'total_tx_var': total_tx_var,
            'total_tx_autocorr': total_tx_autocorr
        }
    
    
@dataclass
class AnomalyResult:
    mcc: int
    mnc: str
    rat: int
    bound_type: int
    score: float
    is_anomaly: int
    


    
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