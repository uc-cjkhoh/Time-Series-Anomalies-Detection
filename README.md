# Time-Series-Anomalies-Detection

## Overview

This project is designed to detect anomalies in time series data, such as transaction counts, using robust statistical and machine learning techniques. The pipeline is implemented in **Python 3.9.18** and is intended to run in **offline mode**. The results are returned as pandas Series and can be ingested into HDFS for further analysis.

---

## Features

- **STL Decomposition**: Separates each time series into trend, seasonal, and residual components.
- **MAD-based Z-Score Detection**: Identifies contextual anomalies using robust statistics.
- **Threshold-Based Detection**: Detects point anomalies in trend and seasonal components.
- **Density and Clustering Filtering**: Removes isolated anomalies and identifies clusters of contextual anomalies.
- **Feature Engineering**: Generates multiple features for downstream analysis or modeling.
- **PySpark Integration**: Supports large-scale data processing and HDFS ingestion.

---

## Project Structure

```
.
├── main_ml.py           # Main pipeline script
├── lib/
│   ├── model.py         # Anomaly detection models and logic
│   ├── util.py          # Utility functions (strength, density, quantiles, etc.)
│   └── dataset.py       # Data loading and preprocessing
└── query/
    └── succ_rate.txt    # SQL query template for data extraction
```

---

## How It Works

### 1. Data Loading

- Data is loaded from SQL queries using the `Dataset` class in `lib/dataset.py`.
- The data is grouped by country (MCC), RAT type, and bound type.

### 2. Preprocessing

- Converts date columns to datetime.
- Computes derived columns such as `failed_count`.

### 3. STL Decomposition

- For each target column (e.g., `count`, `failed_count`), the series is decomposed into:
  - **Trend**
  - **Seasonal**
  - **Residual**

### 4. Anomaly Detection

- **Contextual Anomalies**: Detected on the residual using a MAD-based Z-score and adaptive quantile thresholds.
- **Point Anomalies**: Detected on trend and seasonal components using rolling window statistics and thresholding.
- **Density/Clustering Filtering**: Optionally uses density or DBSCAN clustering to filter out isolated anomalies and identify true contextual clusters.

### 5. Feature Engineering

- Generates features such as:
  - Raw counts
  - Residuals
  - Quantile ranks
  - Expected values
  - Engineered combinations of trend/seasonal components

### 6. Output and Ingestion

- Results are saved as pandas DataFrames and ingested into HDFS in parquet format using PySpark.

---

## How to Run

1. **Install Requirements**

   Make sure you have Python 3.9.18 and the required libraries:
   - pandas
   - numpy
   - statsmodels
   - scikit-learn
   - pyspark
   - tqdm

2. **Run the Main Script**

   ```bash
   python main_ml.py
   ```

   This will:
   - Load data from SQL
   - Process each group (by country, RAT, bound type)
   - Detect anomalies and save results to HDFS

---

## Configuration

Edit the `Configuration` class in `main_ml.py` to:
- Set the window size for STL decomposition
- Adjust anomaly detection thresholds (`two_sigma`, `three_sigma`)
- Specify target countries and SQL file paths

---

## Customization

- **To change anomaly detection logic:**  
  Edit or extend functions in `lib/model.py`.
- **To add new features:**  
  Modify the feature engineering section in `main_ml.py`.
- **To tune density/clustering:**  
  Adjust parameters in `util.py` or `model.py` as needed.

---

## Output

- The final output is a pandas DataFrame with anomaly indicators and features.
- Data is saved to HDFS in parquet format, partitioned by model, year, month, date, and bound type.

---

## Notes

- All anomaly detection results are returned as pandas Series.
- The project is designed to run offline and does not require internet access.
- For best results, tune the window size and thresholds based on your data characteristics.

--- 