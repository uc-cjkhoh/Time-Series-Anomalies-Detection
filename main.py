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


@task(name='Feature Engineering', cache_policy=NO_CACHE)
def feature_engineering(data: pd.Series):
    fe = FeatureEngineer(data)
    
    return {
        'mavg_1': fe.mavg(1),
        'mavg_7': fe.mavg(7),
        'mavg_14': fe.mavg(14),
        'mavg_28': fe.mavg(28),

    }
    

@flow(name='Time Series Anomaly Detection')
def main():
    logger = get_run_logger()
    
    logger.info('Get Configuration')
    config = get_config()
    
    logger.info('Get Query')
    query = setup_environment(config)
    
    logger.info('Get Data')
    impala_db = Impala(host=config['host'], port=config['port'])
    impala_db.cur.execute(query)
    
    column_names = ['dt', 'mcc_mnc', 'rat_type', 'par_bound_type', 'success_count', 'total_count']
    data = pd.DataFrame(impala_db.cur.fetchall(), columns=column_names)
    data['dt'] = pd.to_datetime(data['dt'])
    data.set_index('dt')
     
    # filter by mcc, bound type, rat if more than one included
    distinct_mcc_mnc = np.sort(data['mcc_mnc'].unique())
    distinct_rat_type = np.sort(data['rat_type'].unique())
    distinct_bound_type = np.sort(data['par_bound_type'].unique())
    
    # feature engineering on all data 
    data['fail_count'] = data['total_count'] - data['success_count']
    data['dayofweek'] = data['dt'].dt.dayofweek
     
    # find subdata group by distinct mcc, mnc, bound type, and rat type
    for mcc_mnc in distinct_mcc_mnc:
        for bound in distinct_bound_type:
            for rat in distinct_rat_type:
                subdata = data[(data['mcc_mnc'] == mcc_mnc) & (data['par_bound_type'] == bound) & (data['rat_type'] == rat)]
                
                print(subdata.head())
                # feature engineering on subdata
                # anomaly detection


if __name__ == '__main__':
    main()