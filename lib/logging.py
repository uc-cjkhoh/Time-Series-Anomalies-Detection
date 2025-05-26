def logging(message, filepath='anomaly_log.txt'):
    """
    Log the current process to a file.
    
    Parameters:
    - message: The message to log.
    - filepath: The path to the log file.
    
    Returns:
    - None
    """
    try:
        with open(filepath, 'a') as log_file:
            log_file.write(message)
            log_file.write('\n')
            
        log_file.close()
            
    except Exception as e:
        print(f"Error: {e}")    
