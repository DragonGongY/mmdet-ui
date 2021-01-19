import logging
import os
import time

def get_logger(log_path = 'log_path'):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    timer = time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime())
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s]   %(asctime)s    %(message)s')
    txthandle = logging.FileHandler((log_path + '/' + timer + 'log.txt'))
    txthandle.setFormatter(formatter)
    logger.addHandler(txthandle)
    return logger