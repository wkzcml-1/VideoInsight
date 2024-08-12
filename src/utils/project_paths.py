import os

# Root directory of the project
# _file_ : Project/src/utils/project_paths.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# checkpoints directory
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

# data directory
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# logs directory
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# config file path
CONFIG_FILE = os.path.join(PROJECT_ROOT, 'config.yaml')

if __name__ == '__main__':
    print("PROJECT_ROOT", PROJECT_ROOT)
    print("CHECKPOINTS_DIR", CHECKPOINTS_DIR)
    print("DATA_DIR", DATA_DIR)
    print("LOGS_DIR", LOGS_DIR)
    print("CONFIG_FILE", CONFIG_FILE)