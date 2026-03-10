import pandas as pd
import numpy as np

from prefect import flow, task, get_run_logger
from prefect.cache_policies import NO_CACHE

from src.config.loader import get_config
from src.database.loader import Impala
from src.feature_engineering.feature_creation import FeatureEngineer


@task(name='Setup Environment', cache_policy=NO_CACHE)
def setup_environment(config: dict):
    logger = get_run_logger()
    
    try:
        with open(config['query_file'], 'r') as file:
            query = file.read()  
        return query

    except FileNotFoundError as err:
        logger.error(f'Error occurs: {err}', exc_info=True)  
        raise


@task(name='Check Data Patterns', cache_policy=NO_CACHE)
def has_patterns(data: pd.Series):
    pass
    

@task(name='Feature Engineering', cache_policy=NO_CACHE)
def feature_engineering(data: pd.Series, name: str, num_of_points_per_day: int):
    fe = FeatureEngineer(data)
    trend, seasonal, residual = fe.decomposition(period=num_of_points_per_day)
    power_ratio = fe.fourier_transforms(sampling_interval=num_of_points_per_day)
    
    return pd.DataFrame({
        # moving average
        f'{name}_mavg_per_unit': fe.mavg(1),
        f'{name}_mavg_per_hour': fe.mavg(num_of_points_per_day // 12),
        f'{name}_mavg_per_day': fe.mavg(num_of_points_per_day),
        f'{name}_mavg_week': fe.mavg(num_of_points_per_day * 7),
        
        # autocorrelation
        f'{name}_autocorr': fe.autocorrelation(nlags=len(data)),
        
        # # partial-autocorrelation
        # f'{name}_partial_autocorr': fe.partial_autocorrelation(nlags=len(data)),
        
        # fourier transforms
        f'{name}_fft': power_ratio,
        
        # decomposition
        f'{name}_trend': trend,
        f'{name}_seasonal': seasonal,
        f'{name}_resid': residual
    })
    

@flow(name='Time Series Anomaly Detection')
def main():
    logger = get_run_logger()
    
    # 1. Get system configuration
    logger.info('Get Configuration')
    config = get_config()
    
    # 2. Create required components
    logger.info('Get Query')
    query = setup_environment(config)
    
    # 3. Retrieve data through Impala
    logger.info('Get Data')
    impala_db = Impala(host=config['host'], port=config['port'])
    impala_db.cur.execute(query)

    column_names = ['dt', 'mcc_mnc', 'rat_type', 'par_bound_type', 'success_count', 'total_count']
    data = pd.DataFrame(impala_db.cur.fetchall(), columns=column_names)
    data['dt'] = pd.to_datetime(data['dt'])
    data.set_index('dt')

    # 4. Get distinct mcc, mnc, rat type, bound type
    distinct_mcc_mnc = np.sort(data['mcc_mnc'].unique())
    distinct_rat_type = np.sort(data['rat_type'].unique())
    distinct_bound_type = np.sort(data['par_bound_type'].unique())
    
    # 5. Feature engineering in general
    data['fail_count'] = data['total_count'] - data['success_count']
    data['dayofweek'] = data['dt'].dt.dayofweek
     
    # 6. Find subdata group by distinct mcc, mnc, bound type, and rat type
    for mcc_mnc in distinct_mcc_mnc:
        for bound in distinct_bound_type:
            for rat in distinct_rat_type:
                subdata = data[
                    (data['mcc_mnc'] == mcc_mnc) 
                    & (data['par_bound_type'] == bound) 
                    & (data['rat_type'] == rat)
                ]
                
                if not len(subdata) > 0:
                    continue
                
                # feature engineering on subdata
                for curr_col in config['columns_to_engineer']:
                    curr_col_features = feature_engineering(subdata[curr_col], name=curr_col, num_of_points_per_day=288)
                    subdata = pd.concat([subdata, curr_col_features], axis=1)
                    
                
                print(subdata.head())    
                
                # check if the subdata has pattern(s)
                # if not has_patterns(subdata):
                #     continue
                
                # anomaly detection


if __name__ == '__main__':
    main()