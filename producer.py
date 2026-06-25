# If TCP packet was received, use the following code:
# ----------------------------------------------------
# import socket
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_socket.connect(('127.0.0.1', 65432))
# data = client_socket.recv(1024)
# row = json.loads(data.decode('utf-8'))
# client_socket.close()

import json
import glob
import socket 
import asyncio
import pandas as pd

from typing import Any
from collections.abc import Hashable
from confluent_kafka import Producer


TOPIC = 'roaming_tx_test'

PRODUCER_CONFIG = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': socket.gethostname(),
    'acks': 'all',
    'linger.ms': 10,
    'compression.type': 'snappy',
}

CSV_DTYPES: dict[Hashable, Any] = {
    'trans_iid':            str,
    'trans_eid':            str,
    'mme_host':             str,
    'hss':                  str,
    'status':               str,
    'mnc':                  str,
    'mnc_ref':              str,
    'mnc_ref_primary':      str,
    'msc_mnc_ref':          str,
    'subs_type':            str,
    'rate_plan':            str,
    'lang_code':            str,
    'service_barring':      str,
    'cs_prg_code':          str,
    'principal_rate_plan':  str,
    'subscriber_status':    str,
    'roaming_pass_id':      str,
}



def delivery_report(err, msg):
    if err is not None:
        print(f'Delivery failed | key={msg.key()} | {err}')
    else:
        print(
            f'Delivered | topic={msg.topic()} '
            f'partition=[{msg.partition()}] offset={msg.offset()} '
            f'key={msg.key().decode()}'
        )


async def dynamic_subscribe(producer: Producer):
    data_list = glob.glob('./data/*_transaction.csv')
    
    data = pd.concat(
        (pd.read_csv(file, dtype=CSV_DTYPES) for file in data_list),
        ignore_index=True
    ).sort_values(by='module_dt', ascending=True)

    for row in data.to_dict(orient='records'):
        # Key = mcc_ref-mnc_ref ensures same operator → same partition → same consumer
        key = f"{row.get('mcc_ref', '')}-{row.get('mnc_ref', '')}-{row.get('rat_type', '')}-{row.get('bound_type', '')}"

        producer.produce(
            topic    = TOPIC,
            key      = key.encode('utf-8'),
            value    = json.dumps(row).encode('utf-8'),
            callback = delivery_report,
        )

        # Drain delivery callbacks without blocking the event loop
        producer.poll(0)

        await asyncio.sleep(0.005)

    # Block until all in-flight messages are acknowledged before exit
    remaining = producer.flush(timeout=30)
    if remaining > 0:
        print(f'{remaining} messages were NOT delivered after flush timeout')


if __name__ == '__main__':
    producer = Producer(PRODUCER_CONFIG)
    asyncio.run(dynamic_subscribe(producer))