import os
import logging
from project_paths import LOGS_DIR, DEBUG_DIR
from load_config import load_config

# load config file
config = load_config()

# logging:
#   log_file: video_insight.log
#   log_level: DEBUG
#   console_output: false

log_path = os.path.join(LOGS_DIR, config['logging']['log_file'])
log_level = getattr(logging, config['logging']['log_level'].upper())
console_output = config['logging']['console_output']
clear_old_logs = config['logging']['clear_old_logs']

# create logs directory if it does not exist
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

if clear_old_logs:
    # clear old log file
    if os.path.exists(log_path):
        os.remove(log_path)
    # clear old debug files
    if os.path.exists(DEBUG_DIR):
        # delete all files and directory in the directory
        for file in os.listdir(DEBUG_DIR):
            file_path = os.path.join(DEBUG_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(e)


# logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# file handler
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(log_level)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level) 
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# add handlers to logger
logger.addHandler(file_handler)
if console_output:
    logger.addHandler(console_handler)

## test
if __name__ == '__main__':
    print("log_path", log_path)
    print("log_level", log_level)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")