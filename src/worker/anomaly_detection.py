import queue
import threading

from river import time_series
from river import preprocessing
from river import anomaly
from river import linear_model
from river import optim

from src.processor.stream_state_processor import StatsReport


def perform_detection(work_queue: queue.Queue, stop_event: threading.Event, period: int = 288):
    
    while not stop_event.is_set() or not work_queue.empty():
        try:
            report: StatsReport = work_queue.get()
        except queue.Empty:
            continue
        
        try:
            print(report.get_report()) 
        finally:
            work_queue.task_done()    
            
    # predictive_model = time_series.SNARIMAX(
    #     p=period,
    #     d=1,
    #     q=period,
    #     m=period,
    #     sd=1,
    #     regressor=(
    #         preprocessing.StandardScaler()
    #         | linear_model.LinearRegression(
    #             optimizer=optim.SGD(0.005),
    #         )
    #     ),
    # )

    # PAD = anomaly.PredictiveAnomalyDetection(
    #     predictive_model,
    #     horizon=1,
    #     n_std=3.5,
    #     warmup_period=15
    # )

    # scores = []

    # score = PAD.score_one(None, input) # type: ignore
    # PAD.learn_one(None, input)
    # scores.append(score)

    # print(scores[-1])