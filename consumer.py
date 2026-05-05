import json
import signal
import queue
import threading

from datetime import datetime
from confluent_kafka import Consumer, KafkaError, KafkaException
from src.processor.stream_state_processor import StatsReport 
from src.worker.anomaly_detection import perform_detection


TOPIC = 'roaming_tx_test3'

CONSUMER_CONFIG = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'roaming-monitor-group',
    'auto.offset.reset': 'earliest',        
    'enable.auto.commit': False,            
}
 

def one_minute_interval(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)
 
 
def main():
    # workers shared state
    work_queue = queue.Queue(maxsize=10000)
    stats_report = {}
    stop_event = threading.Event()
    
    # create worker queue for anomaly detection
    worker = threading.Thread(
        target=perform_detection, 
        args=(work_queue, stop_event, 288),
        daemon=True,
        name='anomaly_detection',
    )
    
    worker.start()
    
    # Initialize Consumer
    consumer = Consumer(CONSUMER_CONFIG)
    
    # Subscribe to [TOPIC] tunnel
    consumer.subscribe([TOPIC])

    try:
        while True:
            # poll(1.0) = wait up to 1 second for a message
            msg = consumer.poll(1.0) 
            
            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF: # type: ignore
                    continue
                
                raise KafkaException(msg.error())

            # msg.value() returns raw bytes — decode once here
            try:
                value = json.loads(msg.value().decode('utf-8')) # type: ignore
                tx_dt = datetime.fromtimestamp(value['module_dt'] // 1000)  
                tx_mcc = value['mcc_ref']
                tx_mnc = value['mnc_ref']
                tx_rat = value['rat_type']
                tx_bound_type = value['bound_type']
                bucket_dt = one_minute_interval(tx_dt)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f'Skipping malformed message: {e}')
                continue
 
            # filtering statements
            if str(value.get('op_code')) not in {'2', '23', '316'}:
                continue
 
            group_id = f'{tx_mcc}-{tx_mnc}-{tx_rat}-{tx_bound_type}'
            
            # if not exists in report dict, create new report with new group_id
            if group_id not in stats_report:    
                stats_report[group_id] = StatsReport(tx_mcc, tx_mnc, tx_rat, tx_bound_type)

            # action point
            if stats_report[group_id].get_dt() != bucket_dt:
                report = StatsReport(
                    mcc = stats_report[group_id].get_mcc(),
                    mnc = stats_report[group_id].get_mnc(),
                    rat = stats_report[group_id].get_rat(),
                    bound_type = stats_report[group_id].get_bound_type(),
                    dt = stats_report[group_id].get_dt(),
                    tx_succ_count = stats_report[group_id].get_tx_succ_count(),
                    tx_total_count = stats_report[group_id].get_tx_total_count()
                )
                
                stats_report[group_id].reset_tx(bucket_dt)
                
                try:
                    work_queue.put_nowait(report)
                except queue.Full:
                    print(f'[WARN] queue full - dropping input data for {group_id}')

            tx_status = int(value.get('status_type') == 12 and str(value.get('status')) in {'1', '2', '3', '4', '5'})

            stats_report[group_id].record_tx(tx_status)
 
            consumer.commit(asynchronous=True)
            

    finally:
        worker.join(timeout=10)
        consumer.commit(asynchronous=False)
        consumer.close()


if __name__ == '__main__':
    main()