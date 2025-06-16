from statsmodels.tsa.seasonal import STL
from pyspark.sql import SparkSession
from pyspark.sql.functions import col   
from tqdm import tqdm
 
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
           
        # window size for one hours 
        self.window_size = 288
        
        self.two_sigma = 1.96
        
        self.three_sigma = 2.58
        
        self.must_have_features = ['dt', 'tx_hour', 'count', 'mcc_mnc', 'rat_type', 'par_year', 'par_month', 'par_date', 'par_bound_type']
        
        # self.target_countries = [505,528,456,460,454,404,510,440,530,515,420,525,450,466,520,286,424,234,310,452]
        self.target_countries = [234, 505, 525] 
    
    def import_python(self):
        # import all custom python modules
        for python_file in self.customPythonModule:
            self.spark.sparkContext.addPyFile(python_file)

              
def read_sql_from_file(file_path, mcc): 
    """
    Read or import SQL Query from a text file.

    Args:
        file_path (str): the path to the Query file
        mcc (int): Country code insert into the query

    Returns:
        str: The SQL query
    """
    
    with open(file_path, 'r') as file:
        sql_query = file.read()
    return sql_query.format(mcc, mcc)
 

def insert_into_hdfs(spark_session, data):
    """
    Insert data into HDFS in parquet format.

    Args:
        spark_session (object): Spark session to use for writing data
        data (Dataframe): the data to be inserted into HDFS
    """
    
    spark_df = spark_session.createDataFrame(data) 

    spark_df = spark_df.withColumn("rat_type", col("rat_type").cast("int"))
    spark_df = spark_df.withColumn("dt", col("dt").cast("timestamp")) 
    spark_df = spark_df.withColumn("success_rate", col("success_rate").cast("decimal(20, 6)"))
    
    spark_df = spark_df.withColumn("total_count", col("total_count").cast("int")) 
    spark_df = spark_df.withColumn("feature_1", col("feature_1").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_2", col("feature_2").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_3", col("feature_3").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_4", col("feature_4").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_5", col("feature_5").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_6", col("feature_6").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_7", col("feature_7").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_8", col("feature_8").cast("decimal(20, 6)"))
      
    try:    
        spark_df.write.mode('append').partitionBy('par_model', 'par_year', 'par_month', 'par_date', 'par_bound_type').parquet("hdfs://hadoopha/roam352_report/data/report/data_anomaly_2")
    except Exception as e: 
        print(f"Error: {e}")


def execute_detection(spark_session, data, configuration): 
    """
    The main function to execute the anomaly detection process

    Args:
        spark_session (object): Spark session to use for processinng data
        data (dataframe): the data to be processed 
        configuration (object): user defined configuration or settings
    """
    
    # convert to datetime
    data['dt'] = pd.to_datetime(data['dt'])
    
    # find failed transaction count
    data['failed_count'] = data['total_count'] - data['count']
      
    feature_to_target = ['count', 'failed_count']   
    for column in feature_to_target:
        stl = STL(data[column], period=configuration.window_size, seasonal_deg=0, trend_deg=0, low_pass_deg=0, robust=True).fit()
        data[column + '_residual'] = stl.resid
        data[column + '_seasonal'] = stl.seasonal
        data[column + '_trend'] = stl.trend
        
        data[column + '_seasonality_label'] = seasonality_label = util.check_seasonality(data[column], period=configuration.window_size)
        data[column + '_trend_label'] = trend_label = util.check_trend(data[column], period=configuration.window_size)
          
        # point anomalies detection
        data[column + '_point_anomalies'] = point_abnormal = \
            model.detect_extreme_value(
                data[column].copy(),
                configuration.three_sigma, # higher sigma to capture more extreme value  
                configuration.window_size
            ) 
        
        # contextual anomalies detection 
        contextual_anomalies = model.detect_residual_outlier(
            data[column].copy(),
            configuration.two_sigma,
            configuration.window_size,
            trend_label,
            seasonality_label
        )  
        
        # remove anomalies not in dense
        # contextual_anomalies = util.anomaly_density(
        #     contextual_anomalies,
        #     configuration.window_size,
        #     threshold=0.1
        # )
          
        data[column + '_final_result'] = point_abnormal + contextual_anomalies
     
    
    data['par_model'] = 'ENSEMBLE_MODEL_TEST'
    
    data['is_outlier'] = data['count_final_result'].astype(str)
    
    data['feature_1'] = data['count']
    
    data['feature_2'] = data['failed_count']
     
    data['feature_3'] = data['failed_count_final_result']
    
    data['feature_4'] = data['count_trend']
    
    data['feature_5'] = data['count_seasonal']
    
    data['feature_6'] = data['count_residual']
    
    data['feature_7'] = data['failed_count_trend']
    
    data['feature_8'] = data['failed_count_seasonal']
    
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
    ][configuration.window_size:]  # skip the first 24 hours of data
     
    insert_into_hdfs(spark_session, data_to_ingest)
         
      
def start_action(spark_session, data_source, configuration):
    """
    Preprocess the data and group the into subset based on mcc, mnc, rat type and bound type.

    Args:
        spark_session (object): Spark session initiated 
        data_source (dataframe): data extracted from SQL query
        configuration (object): user fdefined configuration or settings
    """
    
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
                    query=sql_query   
                ) 
                
                start_action(spark_session, data, configuration)
                            
            except Exception as e: 
                print(f"Error: {e}")
                    
    spark_session.stop()