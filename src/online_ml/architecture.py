import queue
import threading
import pickle
import math
import os

from sqlalchemy.engine import URL
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from src.classes.report import ReportSnapshot, FeaturesExtraction, AnomalyReport



CHECKPOINT_DIR = "./model_checkpoints"
SAVE_EVERY_N  = 12



def _checkpoint_path(id: str, obj_type: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{id}__{obj_type}.pkl")


def _save_state(id: str, features: FeaturesExtraction, anomaly_report: AnomalyReport):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(_checkpoint_path(id, "features"), "wb") as f:
        pickle.dump(features, f)
    with open(_checkpoint_path(id, "anomaly"), "wb") as f:
        pickle.dump(anomaly_report, f)


def _load_state(id: str, window_size: int):
    feat_path = _checkpoint_path(id, "features")
    anom_path = _checkpoint_path(id, "anomaly")
    
    if os.path.exists(feat_path) and os.path.exists(anom_path):
        with open(feat_path, "rb") as f:
            features = pickle.load(f)
        with open(anom_path, "rb") as f:
            anomaly_report = pickle.load(f)
        print(f"[{id}] Restored model state from checkpoint.")
        return features, anomaly_report
    
    return FeaturesExtraction(window_size), AnomalyReport(window_size=window_size)
 
 
def detection_pipeline(work_queue: queue.Queue, stop_event: threading.Event, window_size: int, anomaly_threshold: float): 
    all_group_features = {} 
    all_group_anomaly_result = {}
    update_counts = {}
    
    engine = create_engine(
        URL.create(
            drivername='mysql+pymysql',
            host='10.168.51.196',
            port=3306,
            username='unified',
            password='unified',
            database='test_cj'
        )
    )
    
    while not stop_event.is_set() or not work_queue.empty():
        try:
            current_report: ReportSnapshot = work_queue.get()
        except queue.Empty:
            continue
        
        try:  
            id = current_report.report_id 
            
            # Load from checkpoint on first encounter of this id
            if id not in all_group_features:
                all_group_features[id], all_group_anomaly_result[id] = _load_state(id, window_size)
                update_counts[id] = 0
            
            # Update online report statistic
            current_id_features = all_group_features[id].update_and_extract(current_report)
            
            # Update model and score
            all_group_anomaly_result[id].update_model(current_id_features)
            anomaly_score = all_group_anomaly_result[id].get_anomalies_score(current_id_features)
            
            # Periodic checkpoint save
            update_counts[id] += 1
            if update_counts[id] % SAVE_EVERY_N == 0:
                _save_state(id, all_group_features[id], all_group_anomaly_result[id])
                print(f"[{id}] Checkpoint saved at update {update_counts[id]}.")
            
            data = {
                'mcc': current_report.mcc,
                'mnc': 'Unknown' if math.isnan(float(current_report.mnc)) else current_report.mnc,
                'rat_type': current_report.rat,
                'bound_type': current_report.bound_type,
                'tx_dt': current_report.dt,
                
                'succ_tx_count': current_report.tx_succ_count,
                'succ_tx_ma': current_id_features['succ_tx_ma'],
                'succ_tx_mvar': current_id_features['succ_tx_mvar'],  
                'succ_tx_median': current_id_features['succ_tx_median'],
                'succ_count_z_score': current_id_features['succ_count_z_score'],
                
                'total_tx_count': current_report.tx_total_count,
                'total_tx_ma': current_id_features['total_tx_ma'],
                'total_tx_mvar': current_id_features['total_tx_mvar'],
                'total_tx_median': current_id_features['total_tx_median'],
                'total_count_z_score': current_id_features['total_count_z_score'],
                
                'is_anomaly': (anomaly_score > anomaly_threshold),
                'anomaly_score': anomaly_score
            }
            
            try:
                print(f'Insert {data} into database')
                with engine.begin() as conn:
                    conn.execute(
                        text(
                            """
                            INSERT INTO real_time_anomaly_detection_2
                                (mcc, mnc, rat_type, bound_type, tx_dt, succ_tx_count, succ_tx_ma, succ_tx_mvar, succ_tx_median, succ_count_z_score, total_tx_count, total_tx_ma, total_tx_mvar, total_tx_median, total_count_z_score, is_anomaly, anomaly_score)
                            VALUES
                                (:mcc, :mnc, :rat_type, :bound_type, :tx_dt, :succ_tx_count, :succ_tx_ma, :succ_tx_mvar, :succ_tx_median, :succ_count_z_score, :total_tx_count, :total_tx_ma, :total_tx_mvar, :total_tx_median, :total_count_z_score, :is_anomaly, :anomaly_score)
                            ON DUPLICATE KEY UPDATE
                                succ_tx_count = VALUES(succ_tx_count),
                                succ_tx_ma = VALUES(succ_tx_ma),
                                succ_tx_mvar = VALUES(succ_tx_mvar),
                                succ_tx_median = VALUES(succ_tx_median),
                                succ_count_z_score = VALUES(succ_count_z_score),
                                total_tx_count = VALUES(total_tx_count),
                                total_tx_ma = VALUES(total_tx_ma),
                                total_tx_mvar = VALUES(total_tx_mvar),
                                total_tx_median = VALUES(total_tx_median),
                                total_count_z_score = VALUES(total_count_z_score),
                                is_anomaly = VALUES(is_anomaly),
                                anomaly_score = VALUES(anomaly_score)
                            """
                        ),
                        data
                    )
            except ProgrammingError as e:
                print(e)
        
        finally:
            work_queue.task_done()