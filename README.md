# Time-Series Anomaly Detection for Roaming Transactions

A real-time anomaly detection system for monitoring cellular roaming transactions using Apache Kafka and online learning algorithms.

## Project Overview

This system processes transaction streams from multiple cellular operators, aggregating statistics by operator (MCC), network code (MNC), radio access technology (RAT), and traffic direction (bound type). It detects anomalies in transaction success rates using online learning models that require no historical data storage.

**Key Features:**
- Real-time Kafka-based transaction streaming
- Stateful aggregation into configurable time buckets (default: 1 minute)
- Online anomaly detection using River library
- Multi-dimensional grouping (MCC-MNC-RAT-Bound_Type)
- Thread-safe async architecture with separate consumer and detection workers
- Support for Impala database integration for historical data

## Architecture

```
┌──────────────────────────┐
│  Impala Database         │
│  (historical data)       │
└──────────────────────────┘
         ↓
  get_transaction.py
         ↓
┌──────────────────────────┐
│  transaction.csv         │
└──────────────────────────┘
         ↓
  producer.py (CSV → Kafka)
         ↓
┌──────────────────────────┐
│  Apache Kafka Topic      │
│  roaming_tx_test3        │
└──────────────────────────┘
         ↓
┌──────────────────────────────────┐
│  consumer.py                     │
│  (Stream Processor)              │
│  ├─ Filters by op_code           │
│  ├─ Groups by MCC-MNC-RAT-Type   │
│  ├─ Aggregates 1-min stats       │
│  └─ Queues StatsReport objects   │
└──────────────────────────────────┘
         ↓
┌──────────────────────────┐
│  work_queue              │
│  (Max 10,000 items)      │
└──────────────────────────┘
         ↓
┌──────────────────────────────────┐
│  anomaly_detection.py            │
│  (Worker Thread)                 │
│  └─ Apply online learning models │
└──────────────────────────────────┘
         ↓
┌──────────────────────────┐
│  real_time_plot.py       │
│  (Optional visualization)│
└──────────────────────────┘
```

**Components:**

- **producer.py** - Reads transaction CSV, publishes to Kafka with operator-based partitioning
- **consumer.py** - Subscribes to Kafka topic, filters and groups transactions, manages stats state
- **stream_state_processor.py** - `StatsReport` class for tracking transaction statistics
- **anomaly_detection.py** - Worker thread processing statistics with online learning models
- **real_time_plot.py** - Matplotlib-based real-time visualization (optional)
- **get_transaction.py** - Impala database query script to fetch historical transactions

## Installation

### 1. Create Virtual Environment

```bash
python -m venv .venv

# Activate
# Windows:
.\.venv\Scripts\Activate.ps1

# macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -e .
```

Or manually:
```bash
pip install pandas>=2.3.3 numpy>=2.4.4 river>=0.23.0 confluent-kafka statsmodels>=0.14.6 impyla>=0.22.0 tqdm>=4.67.3 mlflow>=3.10.1 websockets>=16.0
```

### 3. Set Up Kafka

```bash
# Using Docker (recommended):
docker-compose up -d kafka zookeeper

# Or create topic manually:
kafka-topics.sh --create \
    --topic roaming_tx_test3 \
    --bootstrap-server localhost:9092 \
    --partitions 4 \
    --replication-factor 1
```

## Configuration

### Kafka Producer Config

[producer.py](producer.py#L12-L18)

```python
PRODUCER_CONFIG = {
    'bootstrap.servers': 'localhost:9092',      # Kafka broker
    'client.id': socket.gethostname(),
    'acks': 'all',                              # Wait for all replicas
    'linger.ms': 10,                            # Batch delay
    'compression.type': 'snappy',
}

TOPIC = 'roaming_tx_test3'
```

### Kafka Consumer Config

[consumer.py](consumer.py#L11-L16)

```python
CONSUMER_CONFIG = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'roaming-monitor-group',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False,
}

TOPIC = 'roaming_tx_test3'
```

### Database Config

[get_transaction.py](get_transaction.py#L7-L13)

```python
conn = connect(
    host='10.168.49.12',
    port=21050,
    database='roam352_report_digi',
    user='unified',
    password='unified'
)
```

### Anomaly Detection Parameters

[anomaly_detection.py](src/worker/anomaly_detection.py#L17-L33) (commented template)

```python
period = 288                    # Seasonality period (hourly → 288 data points/day)
n_std = 3.5                     # Anomaly threshold (standard deviations)
warmup_period = 15              # Minimum observations before detection enabled
```

## Project Structure

```
ts_anomaly_detection_test/
├── README.md
├── pyproject.toml              # Dependencies and project config
│
├── producer.py                 # CSV → Kafka
├── consumer.py                 # Kafka → Stats aggregation
├── get_transaction.py          # Impala → CSV
│
├── src/
│   ├── processor/
│   │   ├── __init__.py
│   │   └── stream_state_processor.py    # StatsReport class
│   ├── worker/
│   │   ├── __init__.py
│   │   └── anomaly_detection.py         # Detection worker (threaded)
│   └── visual/
│       ├── __init__.py
│       └── real_time_plot.py            # Matplotlib dashboard
│
├── data/
│   ├── data.csv
│   └── transaction.csv         # Transaction data (output from get_transaction.py)
│
└── .venv/                      # Virtual environment
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.3.3 | Data frame manipulation |
| numpy | ≥2.4.4 | Numerical computing |
| confluent-kafka | latest | Kafka client |
| river | ≥0.23.0 | Online/incremental ML (time series, anomaly detection) |
| statsmodels | ≥0.14.6 | Statistical models |
| impyla | ≥0.22.0 | Impala/Hive connector |
| mlflow | ≥3.10.1 | ML experiment tracking |
| tqdm | ≥4.67.3 | Progress bars |
| websockets | ≥16.0 | WebSocket support |

**Requires Python ≥3.13**

## API Documentation

### `StatsReport` Class

[stream_state_processor.py](src/processor/stream_state_processor.py)

Tracks transaction statistics for a specific operator/RAT/bound_type combination.

#### Constructor

```python
StatsReport(
    mcc: str,
    mnc: str,
    rat: str,
    bound_type: str,
    dt: datetime = None,
    tx_succ_count: int = 0,
    tx_total_count: int = 0
)
```

**Parameters:**
- `mcc` (str): Mobile Country Code (e.g., "505")
- `mnc` (str): Mobile Network Code (e.g., "12")
- `rat` (str): Radio Access Technology (e.g., "LTE", "5G")
- `bound_type` (str): Direction type ("DL" for downlink, "UL" for uplink)
- `dt` (datetime, optional): Bucket datetime for statistics
- `tx_succ_count` (int): Successful transactions (default 0)
- `tx_total_count` (int): Total transactions (default 0)

#### Methods

```python
# Record a transaction
record_tx(tx_status: int) -> None
# Args: tx_status (0=failed, 1=successful)

# Reset counters for new time bucket
reset_tx(dt: datetime) -> None
# Args: dt - new bucket datetime

# Get statistics
get_mcc() -> str
get_mnc() -> str
get_rat() -> str
get_bound_type() -> str
get_dt() -> datetime
get_tx_succ_count() -> int
get_tx_total_count() -> int

# Export report as dictionary
get_report() -> dict
# Returns: {
#   'dt': datetime,
#   'mcc': str,
#   'mnc': str,
#   'rat_type': str,
#   'bound_type': str,
#   'succ_count': int,
#   'total_count': int
# }
```

#### Example Usage

```python
from src.processor.stream_state_processor import StatsReport
from datetime import datetime

# Create report for LTE downlink traffic
report = StatsReport(
    mcc='505',
    mnc='12',
    rat='LTE',
    bound_type='DL'
)

# Record transactions
report.record_tx(1)  # Success
report.record_tx(1)  # Success
report.record_tx(0)  # Failure

# Get statistics
print(f"Success Rate: {report.get_tx_succ_count()} / {report.get_tx_total_count()}")

# Start new time bucket
report.reset_tx(datetime.now().replace(second=0, microsecond=0))

# Export for analysis
stats = report.get_report()
```

### `perform_detection()` Function

[anomaly_detection.py](src/worker/anomaly_detection.py)

Worker function for processing statistics reports and detecting anomalies.

```python
def perform_detection(
    work_queue: queue.Queue,
    stop_event: threading.Event,
    period: int = 288
) -> None
```

**Parameters:**
- `work_queue` (queue.Queue): Queue containing `StatsReport` objects
- `stop_event` (threading.Event): Signal to stop the worker
- `period` (int): Seasonality period for ARIMA models (default 288 for hourly data)

**Behavior:**
- Continuously processes reports from queue
- Applies anomaly detection algorithms
- Runs until `stop_event` is set and queue is empty

#### Example Usage

```python
import queue
import threading
from src.worker.anomaly_detection import perform_detection

# Create queue and stop event
work_queue = queue.Queue(maxsize=10000)
stop_event = threading.Event()

# Start worker thread
worker = threading.Thread(
    target=perform_detection,
    args=(work_queue, stop_event, 288),
    daemon=True,
    name='anomaly_detection'
)
worker.start()

# ... produce reports to work_queue ...

# Stop worker gracefully
stop_event.set()
worker.join(timeout=10)
```

### `RealtimePlotter` Class

[real_time_plot.py](src/visual/real_time_plot.py)

Real-time data visualization with matplotlib.

```python
class RealtimePlotter:
    def __init__(self, max_points: int = 100)
```

**Parameters:**
- `max_points` (int): Maximum data points per series (default 100)

#### Methods

```python
# Update plots with new statistics
update(stats_report: dict) -> None

# Clear and redraw all plots
_draw_plots() -> None
```

#### Example Usage

```python
import matplotlib.pyplot as plt
from src.visual.real_time_plot import RealtimePlotter

plotter = RealtimePlotter(max_points=100)

# In main loop:
while True:
    stats = get_stats_report()
    plotter.update(stats)
    plt.show()
```

## Data Flow

### Transaction Flow

```
Raw Transaction (source)
    ↓
[Consumer.poll()] - Read from Kafka topic
    ↓
[Decode & Parse] - JSON decode, timestamp conversion
    ↓
[Filter op_code] - Only keep codes: 2, 23, 316
    ↓
[Group By] - MCC-MNC-RAT-Bound_Type
    ↓
[Create StatsReport] - Initialize if first time
    ↓
[Check Time Bucket] - If bucket changed, queue report
    ↓
[Record Transaction] - tx_succ_count++, tx_total_count++
    ↓
[Commit Offset] - Asynchronous offset commit
```

### Statistics Report Format

```json
{
  "dt": "2024-02-28 14:32:00",
  "mcc": "505",
  "mnc": "12",
  "rat_type": "LTE",
  "bound_type": "DL",
  "succ_count": 145,
  "total_count": 152
}
```

### CSV Column Mapping

**Source columns** (transaction.csv) → **Target fields** (StatsReport):

| CSV Column | Field | Type | Purpose |
|-----------|-------|------|---------|
| `module_dt` | `dt` | timestamp | Bucket key (1-min interval) |
| `mcc_ref` | `mcc` | string | Mobile Country Code |
| `mnc_ref` | `mnc` | string | Mobile Network Code |
| `rat_type` | `rat` | string | Network technology (LTE, 5G) |
| `bound_type` | `bound_type` | string | Traffic direction (DL/UL) |
| `op_code` | filter | string | Operation type (2, 23, 316 only) |
| `status_type` + `status` | `tx_succ_count` | binary | Success indicator |

### Anomaly Detection Flow (Future)

```
StatsReport (1-min bucket)
    ↓
[time_series.SNARIMAX] - Seasonal ARIMA model
    ↓
[Scale Input] - StandardScaler normalization
    ↓
[Linear Regressor] - SGD optimization
    ↓
[PAD (Predictive Anomaly Detection)]
    ├─ score_one(x) - Compute anomaly score
    ├─ learn_one(x) - Fit model incrementally
    └─ threshold: μ ± 3.5σ
    ↓
[Alert Mechanism] - Trigger alerts if anomaly detected
```

## Monitoring and Troubleshooting

### Common Issues and Solutions

#### 1. Kafka Connection Failed

```
Error: "Failed to connect to localhost:9092"
```

**Solutions:**
```bash
# Check Kafka is running
docker ps | grep kafka
# or
lsof -i :9092

# Verify broker is responsive
python -c "from confluent_kafka import KafkaError; print('Kafka client ready')"

# Check network connectivity
telnet localhost 9092
```

#### 2. Queue Full Error

```
[WARN] queue full - dropping input data for 505-12-LTE-DL
```

**Cause:** Anomaly detection worker cannot keep up with producer

**Solutions:**
```python
# In consumer.py, increase queue size:
work_queue = queue.Queue(maxsize=50000)  # Was 10000

# Or implement batch processing in anomaly_detection.py
# Or optimize detection algorithm
```

#### 3. Consumer Lag Growing

**Monitor lag:**
```bash
# Check consumer group status
kafka-consumer-groups.sh \
    --bootstrap-server localhost:9092 \
    --group roaming-monitor-group \
    --describe
```

**Solutions:**
- Increase number of consumer threads
- Optimize filtering logic
- Increase Kafka partitions

#### 4. Missing Data in Reports

**Check filtering:**
```python
# In consumer.py, verify op_code filter:
if str(value.get('op_code')) not in {'2', '23', '316'}:
    print(f"Filtered out op_code: {value.get('op_code')}")
    continue
```

**Verify CSV input:**
```bash
# Check CSV has required columns
python -c "import pandas as pd; df = pd.read_csv('./data/transaction.csv'); print(df.columns.tolist())"
```

### Logging and Debugging

#### Enable Debug Output

```python
# In consumer.py
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Processing: {group_id}")
logger.info(f"Queued report for {group_id}")
```

#### Monitor Queue Status

```python
# Add periodic monitoring
import threading
import time

def monitor_queue(work_queue):
    while True:
        print(f"Queue size: {work_queue.qsize()}")
        time.sleep(5)

monitor_thread = threading.Thread(target=monitor_queue, args=(work_queue,), daemon=True)
monitor_thread.start()
```

#### Profile Performance

```bash
# Profile memory usage
python -m memory_profiler consumer.py

# Profile execution time
python -m cProfile -s cumtime consumer.py | head -30
```

### Health Checks

#### Producer Health

```bash
# Monitor messages produced
kafka-console-consumer.sh \
    --bootstrap-server localhost:9092 \
    --topic roaming_tx_test3 \
    --max-messages 10
```

#### Consumer Health

```python
# In consumer.py, add metrics:
import time

start_time = time.time()
msg_count = 0

while True:
    msg = consumer.poll(1.0)
    if msg and not msg.error():
        msg_count += 1
        
    if time.time() - start_time >= 60:  # Every minute
        rate = msg_count / 60
        print(f"Processing rate: {rate:.2f} msg/sec")
        msg_count = 0
        start_time = time.time()
```

### Performance Tuning

#### Kafka Producer Optimization

```python
# In producer.py
PRODUCER_CONFIG = {
    'batch.size': 32768,              # Larger batch size
    'linger.ms': 100,                 # Longer linger for batching
    'compression.type': 'snappy',     # Compression
    'acks': 1,                        # Faster acks (if acceptable)
    'buffer.memory': 67108864,        # 64MB buffer
}
```

#### Kafka Consumer Optimization

```python
# In consumer.py
CONSUMER_CONFIG = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'roaming-monitor-group',
    'max.poll.records': 500,          # Fetch more per poll
    'fetch.min.bytes': 1024,          # Min bytes per fetch
    'session.timeout.ms': 30000,      # Session timeout
}
```

#### Anomaly Detection Optimization

```python
# Use batch processing instead of per-report
def perform_detection_optimized(work_queue, stop_event, batch_size=10):
    batch = []
    while not stop_event.is_set() or not work_queue.empty():
        try:
            report = work_queue.get(timeout=1.0)
            batch.append(report)
            
            if len(batch) >= batch_size:
                process_batch(batch)
                batch = []
        except queue.Empty:
            if batch:
                process_batch(batch)
                batch = []
```

## Advanced Topics

### Custom Models

To implement custom anomaly detection models:

1. **Replace SNARIMAX model** in [anomaly_detection.py](src/worker/anomaly_detection.py#L32)
   ```python
   custom_model = MyCustomModel()
   ```

2. **Available River models:**
   - `time_series.SNARIMAX` - Seasonal ARIMA
   - `anomaly.HalfSpaceTrees` - Tree-based detection
   - `anomaly.OneClassSVM` - Support Vector approach
   - Stream-compatible models only

3. **Example custom model:**
   ```python
   from river import linear_model, preprocessing
   
   model = (
       preprocessing.StandardScaler() |
       linear_model.LogisticRegression()
   )
   ```

### Scaling Horizontally

For production deployments:

1. **Multiple Consumer Instances:**
   - Each instance joins same consumer group
   - Kafka automatically distributes partitions
   - Results in parallel processing

   ```bash
   # Terminal 1
   CONSUMER_GROUP=roaming-monitor-group python consumer.py
   
   # Terminal 2
   CONSUMER_GROUP=roaming-monitor-group python consumer.py
   ```

2. **Multiple Producers:**
   - Distribute CSV files across producers
   - Each produces to same topic
   - Load balancing via partitioning

3. **Distributed Anomaly Detection:**
   - Use separate queue per group (sharding)
   - Process in parallel with multiple workers

### Integration with MLflow

Track experiments and models:

```python
import mlflow

mlflow.start_run()
mlflow.log_param("period", 288)
mlflow.log_param("n_std", 3.5)

# Train model
model.learn_one(x, y)

mlflow.log_metric("anomaly_score", score)
mlflow.end_run()
```

### Alerting Integration

Send alerts beyond console output:

```python
# In anomaly_detection.py
def send_alert(report, anomaly_score):
    if anomaly_score > THRESHOLD:
        # Email alert
        send_email(f"Anomaly: {report['mcc']}-{report['mnc']}")
        
        # Slack notification
        send_slack_message(json.dumps(report))
        
        # Database insert
        insert_alert_db(report, anomaly_score)
```

## Contributing

To contribute improvements:

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes following project structure
3. Test with: `python -m pytest tests/`
4. Submit pull request with documentation

## License

Specify your license here.

## Support

For issues or questions:
- Check [Monitoring and Troubleshooting](#monitoring-and-troubleshooting) section
- Review logs and error messages
- Verify Kafka connectivity and data format
- Check dependencies are installed: `pip list | grep -E "pandas|river|confluent"`

## Changelog

### Version 0.1.0 (Current)
- Initial implementation
- Kafka producer/consumer pipeline
- Basic statistics aggregation
- Anomaly detection framework (template)
- Real-time visualization support
