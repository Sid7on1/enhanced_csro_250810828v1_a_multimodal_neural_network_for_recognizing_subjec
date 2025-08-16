import logging
import os
import yaml
from typing import Dict, List, Optional
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
CONFIG_FILE = 'config.yaml'
DEFAULT_CONFIG = {
    'model': {
        'type': 'baseline',
        'pretrained': True
    },
    'data': {
        'path': './data',
        'format': 'csv'
    },
    'training': {
        'batch_size': 32,
        'epochs': 10,
        'learning_rate': 0.001
    }
}

# Define an Enum for model types
class ModelType(Enum):
    BASELINE = 'baseline'
    ATTENTION = 'attention'
    PRETRAINED = 'pretrained'

# Define an abstract base class for config
class Config(ABC):
    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

# Define a concrete config class
class ModelConfig(Config):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_type = ModelType(config['model']['type'])

    def validate(self):
        if self.model_type not in [ModelType.BASELINE, ModelType.ATTENTION, ModelType.PRETRAINED]:
            raise ValueError('Invalid model type')
        if not isinstance(self.config['model']['pretrained'], bool):
            raise ValueError('Invalid pretrained value')

    def load(self):
        logger.info('Loading model config')
        self.model_type = ModelType(self.config['model']['type'])
        self.config['model']['pretrained'] = bool(self.config['model']['pretrained'])

    def save(self):
        logger.info('Saving model config')
        self.config['model']['type'] = self.model_type.value
        self.config['model']['pretrained'] = bool(self.config['model']['pretrained'])

# Define a concrete config class
class DataConfig(Config):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.data_path = config['data']['path']
        self.data_format = config['data']['format']

    def validate(self):
        if not isinstance(self.data_path, str) or not os.path.exists(self.data_path):
            raise ValueError('Invalid data path')
        if self.data_format not in ['csv', 'json']:
            raise ValueError('Invalid data format')

    def load(self):
        logger.info('Loading data config')
        self.data_path = self.config['data']['path']
        self.data_format = self.config['data']['format']

    def save(self):
        logger.info('Saving data config')
        self.config['data']['path'] = self.data_path
        self.config['data']['format'] = self.data_format

# Define a concrete config class
class TrainingConfig(Config):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']
        self.learning_rate = config['training']['learning_rate']

    def validate(self):
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError('Invalid batch size')
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError('Invalid epochs')
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError('Invalid learning rate')

    def load(self):
        logger.info('Loading training config')
        self.batch_size = self.config['training']['batch_size']
        self.epochs = self.config['training']['epochs']
        self.learning_rate = self.config['training']['learning_rate']

    def save(self):
        logger.info('Saving training config')
        self.config['training']['batch_size'] = self.batch_size
        self.config['training']['epochs'] = self.epochs
        self.config['training']['learning_rate'] = self.learning_rate

# Define a config manager class
class ConfigManager:
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.config = DEFAULT_CONFIG

    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.info('Config file not found, using default config')
        except yaml.YAMLError as e:
            logger.error(f'Error loading config: {e}')

    def save_config(self):
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def get_config(self) -> Dict:
        return self.config

    def set_config(self, config: Dict):
        self.config = config
        self.save_config()

# Define a context manager for config loading
@contextmanager
def load_config(config_file: str = CONFIG_FILE):
    config_manager = ConfigManager(config_file)
    config_manager.load_config()
    yield config_manager.get_config()
    config_manager.save_config()

# Define a function to get the config
def get_config() -> Dict:
    with load_config() as config:
        return config

# Define a function to set the config
def set_config(config: Dict):
    config_manager = ConfigManager()
    config_manager.set_config(config)

# Define a function to validate the config
def validate_config(config: Dict):
    model_config = ModelConfig(config['model'])
    data_config = DataConfig(config['data'])
    training_config = TrainingConfig(config['training'])
    model_config.validate()
    data_config.validate()
    training_config.validate()

# Define a function to load the config
def load_config_file(config_file: str = CONFIG_FILE):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Define a function to save the config
def save_config_file(config: Dict, config_file: str = CONFIG_FILE):
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

# Define a function to get the config path
def get_config_path() -> str:
    return os.path.join(os.getcwd(), CONFIG_FILE)

# Define a function to set the config path
def set_config_path(config_path: str):
    global CONFIG_FILE
    CONFIG_FILE = config_path

# Define a function to get the default config
def get_default_config() -> Dict:
    return DEFAULT_CONFIG

# Define a function to set the default config
def set_default_config(config: Dict):
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config