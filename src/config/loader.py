import yaml

from prefect import task, get_run_logger
from prefect.cache_policies import NO_CACHE


class ConfigLoader:
    def __new__(cls):        
        logger = get_run_logger()
        
        try:
            with open('./configs/configs.yaml', 'r') as file:
                return yaml.safe_load(file)
        
        except FileNotFoundError as err:
            logger.error(msg=f'Error occurs: {err}', exc_info=True)
            raise     
            
            
@task(name='', cache_policy=NO_CACHE)
def get_config():
    return ConfigLoader()