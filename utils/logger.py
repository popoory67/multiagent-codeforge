import os
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(name, log_prefix=None, level=logging.DEBUG):
    log_file = log_prefix + f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    # Log files are stored in the "logs" directory
    log_dir = "logs"
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # The maximum of file size is 1MB, and keep 5 backup files 
        file_handler = RotatingFileHandler(log_path, maxBytes=1000000, backupCount=5)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def log_execution_time(func):
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        logger.info(f"Starting {func.__name__} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Finished {func.__name__} at {time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {end_time - start_time:.2f} seconds)")
        return result
    return wrapper