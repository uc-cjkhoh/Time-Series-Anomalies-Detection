import functools
import time

def timeDecorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f'\'{func.__name__}\' started at', time.asctime())
        now = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        end = time.perf_counter()
        print(f'\'{func.__name__}\' ended at', time.asctime())
        
        print(f'\'{func.__name__}\' completed in', round(end - now, 3), 'seconds')
        return result
    return wrapper()


def debugDecorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pass


def logDecorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # key component of logging
        # 1. Log Level
        # 2. Loggers
        # 3. Handlers
        # 4. Formatters
        # 5. Filters
        
        pass