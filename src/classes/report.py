from dataclasses import dataclass
from datetime import datetime
from river import stats, anomaly, preprocessing, compose, utils

         

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



class FeaturesExtraction:
    def __init__(self, window_size):
        self.succ_tx_median = stats.RollingQuantile(q=0.5, window_size=window_size)  
        self.succ_tx_ma = utils.Rolling(stats.Mean(), window_size=window_size)
        self.succ_tx_mvar = utils.Rolling(stats.Var(), window_size=window_size)
        
        self.total_tx_median = stats.RollingQuantile(q=0.5, window_size=window_size) 
        self.total_tx_ma = utils.Rolling(stats.Mean(), window_size=window_size)
        self.total_tx_mvar = utils.Rolling(stats.Var(), window_size=window_size)
        
            
    def update_and_extract(self, snapshot: ReportSnapshot):
        current_succ_count = snapshot.tx_succ_count
        current_total_count = snapshot.tx_total_count
        
        # success transaction count statistic
        # current_succ_tx_median = self.succ_tx_median.get() or current_succ_count
        current_succ_tx_ma = self.succ_tx_ma.get() or current_succ_count
        current_succ_tx_std = (self.succ_tx_mvar.get() or 0) ** 0.5
        
        self.succ_tx_median.update(current_succ_count) # type: ignore  
        self.succ_tx_ma.update(current_succ_count)
        self.succ_tx_mvar.update(current_succ_count)
         
        # total transcation count statistic 
        # current_total_tx_median = self.total_tx_median.get() or current_total_count  
        current_total_tx_ma = self.total_tx_ma.get() or current_total_count
        current_total_tx_std = (self.total_tx_mvar.get() or 0) ** 0.5
        
        self.total_tx_median.update(current_total_count) # type: ignore 
        self.total_tx_ma.update(current_total_count)
        self.total_tx_mvar.update(current_total_count)
         
        return {
            'dt': int(snapshot.dt.timestamp() * 1000),
            'succ_tx_count': snapshot.tx_succ_count,
            'succ_tx_ma': self.succ_tx_ma.get(),
            'succ_tx_mvar': self.succ_tx_mvar.get(),
            'succ_tx_median': self.succ_tx_median.get(),
            'succ_count_z_score': (snapshot.tx_succ_count - current_succ_tx_ma) / (current_succ_tx_std + 1e-9),
            'total_tx_count': snapshot.tx_total_count,
            'total_tx_ma': self.total_tx_ma.get(),
            'total_tx_mvar': self.total_tx_mvar.get(),
            'total_tx_median': self.total_tx_median.get(), 
            'total_count_z_score': (snapshot.tx_total_count - current_total_tx_ma) / (current_total_tx_std + 1e-9)
        }
    


class AnomalyReport:
    def __init__(self, window_size: int):
        self.model =  compose.Pipeline(
            preprocessing.StandardScaler(),
            anomaly.HalfSpaceTrees(window_size=window_size)
        )
    
    def update_model(self, x: dict):
        self.model.learn_one(x)

    def get_anomalies_score(self, x: dict):
        anomaly_score = self.model.score_one(x)
        self.model.learn_one(x)
        
        return anomaly_score
    
    
    
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