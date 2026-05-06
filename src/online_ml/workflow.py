import queue
import threading

from src.classes.report import ReportSnapshot, OnlineMLReport, AnomalyResult


def anomaly_detection(work_queue: queue.Queue, stop_event: threading.Event, window_size): 
    online_reports = {} 
    
    while not stop_event.is_set() or not work_queue.empty():
        try:
            report: ReportSnapshot = work_queue.get()
        except queue.Empty:
            continue
        try:  
            id = report.report_id 
            if id not in online_reports:
                online_reports[id] = OnlineMLReport(window_size)
            
            online_reports[id].update_and_extract(report)
                
            
        finally:
            work_queue.task_done()