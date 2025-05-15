from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from scipy.signal import find_peaks
from tqdm import tqdm

import sys
import logging 
import warnings
import pandas as pd
import numpy as np 

warnings.filterwarnings("ignore")

class Configuration:
    def __init__(self):
        # Initial PySpark Session
        self.spark = SparkSession \
                    .builder \
                    .appName("Anomaly Detection") \
                    .config("spark.yarn.appMasterEnv.JAVA_HOME", "/usr/lib/jvm/jre-11") \
                    .config("spark.sql.legacy.json.allowEmptyString.enabled", value=True) \
                    .enableHiveSupport() \
                    .getOrCreate()
        
        # Define paths to your custom python modules
        self.customPythonModule = [
            '/unified/user/cj/python_module/anomaly_detection_v2/lib/util.py',
            '/unified/user/cj/python_module/anomaly_detection_v2/lib/dataset.py',
            '/unified/user/cj/python_module/anomaly_detection_v2/lib/model.py'
        ]
          
        # Define paths to your text files
        self.sql_files = [
            '/unified/user/cj/python_module/anomaly_detection_v2/query/succ_rate.txt'
        ] 
         
        # how many data point a day. Eg 24 if granularity is hourly 
        self.no_of_loop = 24
        
        # data point for one cycle 
        self.window_size = 12
        
        self.threshold = 3
        
        self.target_column = 'success_rate' # either success_rate or total_count
        
        self.must_have_features = ['dt', 'tx_hour', 'count', 'mcc_mnc', 'rat_type', 'par_year', 'par_month', 'par_date', 'par_bound_type']
        
        # self.target_countries = [505,528,456,460,454,404,510,440,530,515,420,525,450,466,520,286,424,234,310,452]
        self.target_countries = [234,286,424,452,525]
    
    def import_python(self):
        # import all custom python modules
        for python_file in self.customPythonModule:
            self.spark.sparkContext.addPyFile(python_file)

              
def read_sql_from_file(file_path, mcc): 
    with open(file_path, 'r') as file:
        sql_query = file.read()
    return sql_query.format(mcc, mcc)


def execute_sql(spark, sql_query): 
    return spark.sql(sql_query)
  

def insert_into_hdfs(spark_session, data):
    # import into hdfs
    spark_df = spark_session.createDataFrame(data) 

    spark_df = spark_df.withColumn("rat_type", col("rat_type").cast("int"))
    spark_df = spark_df.withColumn("dt", col("dt").cast("timestamp")) 
    spark_df = spark_df.withColumn("success_rate", col("success_rate").cast("decimal(20, 6)"))
    # spark_df = spark_df.withColumn("trend"   , col("trend"   ).cast("decimal(20, 6)"))
    # spark_df = spark_df.withColumn("seasonal", col("seasonal").cast("decimal(20, 6)"))
    # spark_df = spark_df.withColumn("residual", col("residual").cast("decimal(20, 6)"))
    # spark_df = spark_df.withColumn("metric_1", col("metric_1").cast("decimal(20, 6)"))
    # spark_df = spark_df.withColumn("metric_2", col("metric_2").cast("decimal(20, 6)"))
    
    spark_df = spark_df.withColumn("total_count", col("total_count").cast("int")) 
    spark_df = spark_df.withColumn("feature_1", col("feature_1").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_2", col("feature_2").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_3", col("feature_3").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_4", col("feature_4").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_5", col("feature_5").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_6", col("feature_6").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_7", col("feature_7").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_8", col("feature_8").cast("decimal(20, 6)"))
      
    # for col_name in data.columns:
    #     if col_name[-6:] != 'result' and col_name[-11:] != 'lower_bound' and col_name[-11:] != 'upper_bound' and col_name[:11] == 'total_count':
    #         spark_df = spark_df.withColumn(col_name, col(col_name).cast("int"))
    
    # print(spark_df.columns)
    # print(data[24:].head(10).T)
    # spark_df.show(10)

    # sys.exit()
    try:    
        spark_df.write.mode('append').partitionBy('par_model', 'par_year', 'par_month', 'par_date', 'par_bound_type').parquet("hdfs://hadoopha/roam352_report/data/report/data_anomaly_2")
    except Exception as e:
        # util.logging(e)
        print(f"Error: {e}")


def execute_detection(spark_session, data, configuration): 
    logging.info(f"===== Preprocessing Data")     
    
    # convert to datetime
    data['dt'] = pd.to_datetime(data['dt'])
    
    # find failed transaction count
    data['failed_count'] = data['total_count'] - data['count']
    
    # succ_rate_stl = STL(data['success_rate'], period=configuration.window_size * configuration.no_of_loop, seasonal_deg=0, trend_deg=0, low_pass_deg=0, robust=True).fit()
    succ_tx_stl = STL(data['count'], period=configuration.window_size * configuration.no_of_loop, seasonal_deg=0, trend_deg=0, low_pass_deg=0, robust=True).fit()
    fail_tx_stl = STL(data['failed_count'], period=configuration.window_size * configuration.no_of_loop, seasonal_deg=0, trend_deg=0, low_pass_deg=0, robust=True).fit()

    data['success_count_trend'] = succ_tx_stl.trend
    data['success_count_seasonal'] = succ_tx_stl.seasonal
    data['success_count_residual'] = succ_tx_stl.resid
    
    data['failed_count_trend'] = fail_tx_stl.trend
    data['failed_count_seasonal'] = fail_tx_stl.seasonal
    data['failed_count_residual'] = fail_tx_stl.resid 
    
    peaks, _ = find_peaks(fail_tx_stl.seasonal, distance=configuration.window_size * configuration.no_of_loop)
    valleys, _ = find_peaks(-fail_tx_stl.seasonal, distance=configuration.window_size * configuration.no_of_loop)
     
    # does this dataset has seasonality ? 
    
    
    
    # determine the cycle range and label the cycle
    # cycle_range = [tuple(sorted((data['dt'].iloc[start], data['dt'].iloc[end]))) for start, end in zip(valleys, peaks)]
    # print(cycle_range)
    
    # data['cycle_label'] = np.nan
    # for i, (start, end) in enumerate(cycle_range):
    #     data.loc[(data['dt'] >= start) & (data['dt'] <= end), 'cycle_label'] = i 
         
    # print(data['cycle_label'].values)
         
    # sys.exit()
    
    
    # check if the data is messy 
    # rate_std = data['success_rate'].std()
    
    data = data.dropna()
    
        
    
    # 1. Point Anomalies  
    succ_tx_result = model.detect_point_anomalies(data['count'], configuration.threshold, configuration.window_size)
    failed_tx_result = model.detect_point_anomalies(data['failed_count'], configuration.threshold, configuration.window_size)
    
    models = [succ_tx_result, failed_tx_result]
    
    final_result = np.zeros(data.shape[0])
    for i, temp_model in enumerate(models):
        models[i] = np.where(temp_model >= 2, 2^i, 0)
        final_result += models[i]
        
        
        
    # 2. Cotextual Anomalies   
    data['expected_success_value'] = data['success_count_trend'] + data['success_count_seasonal']
    data['expecetd_failed_value'] = data['failed_count_trend'] + data['failed_count_seasonal']
    
    data['failed_count_context'] = model.detect_contextual_anomalies(data[['expecetd_failed_value', 'failed_count_residual']])
    data['success_count_context'] = model.detect_contextual_anomalies(data[['expected_success_value', 'success_count_residual']])
    
    # check if a series of data is seasonaly
    # temp = np.zeros(data.shape[0])
    # acf_result = acf(
    #     data['failed_count_seasonal'].diff().dropna(), 
    #     nlags=data['failed_count'].shape[0] - 1
    # )[1:]
    
    # temp[0: acf_result.shape[0]] = acf_result 
     
    
    # 3. Collective Anomalies
    x = np.arange(0, data.shape[0], 1)
    temp_succ = np.polyfit(x, data['count'], deg=3)
    temp_fail = np.polyfit(x, data['failed_count'], deg=3)
     
    
    data['par_model'] = 'ENSEMBLE_MODEL_V6'
    
    data['is_outlier'] = np.where(final_result > 0, np.vectorize(lambda x: format(int(x), 'b'))(final_result),'0').astype(str)
    
    data['feature_1'] = data['count']
    
    data['feature_2'] = data['failed_count']
    
    data['feature_3'] = data['failed_count_seasonal']
            
    data['feature_4'] = data['failed_count_residual']
    
    data['feature_5'] = data['failed_count_trend']
    
    data['feature_6'] = data['failed_count_context']
    
    data['feature_7'] = np.polyval(temp_fail, x)
    
    data['feature_8'] = np.polyval(temp_succ, x)
    
    data_to_ingest = data[
        [
            'dt', 
            'mcc_mnc', 
            'rat_type',  
            'success_rate', 
            'is_outlier',  
            'total_count',
            'feature_1',
            'feature_2',
            'feature_3', 
            'feature_4', 
            'feature_5', 
            'feature_6', 
            'feature_7', 
            'feature_8',
            'par_model', 
            'par_year', 
            'par_month', 
            'par_date', 
            'par_bound_type'
        ]
    ][configuration.window_size:]
     
    insert_into_hdfs(spark_session, data_to_ingest)
         
      
def start_action(spark_session, data_source, configuration):
    mcc_list = data_source.get_mcc_list()
    bound_list = data_source.get_bound_list()
    rat_list = data_source.get_rat_list() 
 
    for mcc_mnc in tqdm(mcc_list): 
        for bound_type in bound_list:
            for rat_type in rat_list:   
                # get data filtered by mcc_mnc, bound_type, rat_type
                data = data_source.get_data(mcc_mnc, bound_type, rat_type)
                
                if data.empty:
                    print(f"===== Skipping - No Data Found in MCC-MNC: {mcc_mnc}, Bound Type: {bound_type}, RAT Type: {rat_type}") 
                    continue
                
                print(f"===== Now running - MCC-MNC: {mcc_mnc}, Bound Type: {bound_type}, RAT Type: {rat_type}") 
                 
                # data['success_rate_detrended'] = make_stationary(data['success_rate']) 
                # data['total_count_detrended'] = make_stationary(data['total_count'])
                 
                execute_detection(spark_session, data, configuration)
 
                 
if __name__ == "__main__":
    # Initialize SparkSession
    configuration = Configuration()
    configuration.import_python() 
    
    import dataset
    import model
    import util
     
    # import settings
    spark_session = configuration.spark
    sql_files = configuration.sql_files
      
    for file_path in sql_files:
        favorite_mcc = configuration.target_countries

        for mcc in favorite_mcc: 
            # Read the text file into a Spark DataFrame
            sql_query = read_sql_from_file(file_path, mcc) 
            
            try:
                # get all data from sql 
                data = dataset.Dataset(
                    window_size=configuration.window_size, 
                    query=sql_query   
                ) 
                
                start_action(spark_session, data, configuration)
                            
            except Exception as e:
                # util.logging(e)
                print(f"Error: {e}")
                    
    spark_session.stop()
