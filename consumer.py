import json
import queue
import threading

from datetime import datetime
from confluent_kafka import Consumer, KafkaError, KafkaException
from src.classes.report import PeriodicReport, ReportSnapshot
from src.online_ml.architecture import detection_pipeline


# kafka configurations
TOPIC = 'roaming_tx_test'
CONSUMER_CONFIG = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'roaming-monitor-group',
    'auto.offset.reset': 'earliest',        
    'enable.auto.commit': False,            
}
COMMIT_N = 100

# anomaly detection configurations
WINDOW_SIZE=288 
ANOMALY_THRESHOLD=0.9


def one_minute_interval(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)
 
 
def main():  
    work_queue = queue.Queue(maxsize=10000)
    stop_event = threading.Event()
    
    worker = threading.Thread(
        target=detection_pipeline, 
        args=(work_queue, stop_event, WINDOW_SIZE, ANOMALY_THRESHOLD),
        daemon=True,
        name='anomaly_detection',
    ) 
    worker.start()
      
    consumer = Consumer(CONSUMER_CONFIG) 
    consumer.subscribe([TOPIC])
 
    try:
        stats_report = {}
        processed = 0 
    
        while True: 
            msg = consumer.poll(1.0) 
            
            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF: # type: ignore
                    continue
                
                raise KafkaException(msg.error())
 
            value = json.loads(msg.value().decode('utf-8')) # type: ignore 
            
            # DATA FILTERING
            if int(value.get('op_code')) not in [2, 23, 316]:
                continue
            
            tx_dt = datetime.fromtimestamp(value['module_dt'] // 1000)  
            tx_mcc = value['mcc_ref']
            tx_mnc = value['mnc_ref']
            tx_rat = value['rat_type']
            tx_bound_type = value['bound_type']
            
            bucket_dt = one_minute_interval(tx_dt)
            group_id = f'{tx_mcc}-{tx_mnc}-{tx_rat}-{tx_bound_type}'
             
            if group_id not in stats_report:    
                stats_report[group_id] = PeriodicReport(tx_mcc, tx_mnc, tx_rat, tx_bound_type, bucket_dt)
 
            # SEND / MODIFY REPORT
            current_report = stats_report[group_id]
            if stats_report[group_id].dt != bucket_dt:
                report = ReportSnapshot(
                    mcc = current_report.mcc,
                    mnc = current_report.mnc,
                    rat = current_report.rat,
                    bound_type = current_report.bound_type,
                    dt = current_report.dt,
                    tx_succ_count = current_report.tx_succ_count,
                    tx_total_count = current_report.tx_total_count
                )
                
                current_report.reset_tx(bucket_dt)
                
                try:
                    work_queue.put_nowait(report)
                except queue.Full:
                    print(f'[WARN] queue full - dropping input data for {group_id}')

            tx_status = int(value.get('status_type') == 12 and int(value.get('status')) in [1, 2, 3, 4, 5])

            stats_report[group_id].record_tx(tx_status)
 
            processed += 1
            if processed % 100 == 0:
                consumer.commit(asynchronous=True)

    finally:
        worker.join(timeout=10)
        consumer.commit(asynchronous=False)
        consumer.close()


if __name__ == '__main__':
    main()