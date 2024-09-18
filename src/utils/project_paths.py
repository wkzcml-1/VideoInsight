import os
# get date
from datetime import datetime

# Root directory of the project
# _file_ : Project/src/utils/project_paths.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# checkpoints directory
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

# data directory
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# logs directory
# today's date
TODAY = datetime.now().strftime("%Y-%m-%d")
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs', TODAY)

# video store directory
VIDEO_STORE_DIR = os.path.join(DATA_DIR, 'videos', TODAY)
if not os.path.exists(VIDEO_STORE_DIR):
    os.makedirs(VIDEO_STORE_DIR)

# config file path
CONFIG_FILE = os.path.join(PROJECT_ROOT, 'config.yaml')

# debug results directory
DEBUG_DIR = os.path.join(LOGS_DIR, 'debug')

if __name__ == '__main__':
    print("PROJECT_ROOT", PROJECT_ROOT)
    print("CHECKPOINTS_DIR", CHECKPOINTS_DIR)
    print("DATA_DIR", DATA_DIR)
    print("LOGS_DIR", LOGS_DIR)
    print("CONFIG_FILE", CONFIG_FILE)