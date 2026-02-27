import os
import io
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding="utf-8",
    errors="replace",
    line_buffering=True
)

sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer,
    encoding="utf-8",
    errors="replace",
    line_buffering=True
)

# Turns off buffering for all print statements to ensure logs are written immediately
sys.stdout.reconfigure(line_buffering=True)

def setup_logger(name, log_prefix=None, level=logging.DEBUG):
    log_file = log_prefix + f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=1000000,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(level)

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger