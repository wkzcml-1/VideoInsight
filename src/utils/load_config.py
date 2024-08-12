from project_paths import CONFIG_FILE
import yaml

# load config file
with open(CONFIG_FILE, 'r') as file:
    config = yaml.safe_load(file)

def load_config():
    return config