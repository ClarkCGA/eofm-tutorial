import logging
import os

def setup_logger(log_dir, log_name, use_date=True):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{log_name}.log")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if already added
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def progress_reporter(msg, verbose=True, logger=None):
    if verbose:
        print(msg)
    if logger is not None:
        logger.info(msg)
