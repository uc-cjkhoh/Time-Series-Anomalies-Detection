import queue
import threading

from src.classes.report import ReportSnapshot, FeaturesExtraction, AnomalyReport


def detection_pipeline(work_queue: queue.Queue, stop_event: threading.Event, window_size): 
    all_group_features = {} 
    all_group_anomaly_result = {}
    
    while not stop_event.is_set() or not work_queue.empty():
        try:
            current_report: ReportSnapshot = work_queue.get()
        except queue.Empty:
            continue
         
        try:  
            id = current_report.report_id 
            if id not in all_group_features:
                all_group_features[id] = FeaturesExtraction(window_size)
            
            # update online report statistic
            current_id_features = all_group_features[id].update_and_extract(current_report)
            
            # get anomaly detection result
            if id not in all_group_anomaly_result:
                all_group_anomaly_result[id] = AnomalyReport(window_size=window_size)
                
            all_group_anomaly_result[id].update_model(current_id_features)
            anomaly_score = all_group_anomaly_result[id].get_anomalies_score(current_id_features)
            
            print(f'{id}: {anomaly_score}')
            
            # send to database
            
        finally:
            work_queue.task_done()
            