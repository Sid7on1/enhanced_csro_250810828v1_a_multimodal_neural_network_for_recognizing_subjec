import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
DATA_DIR = 'data'
BATCH_SIZE = 32
NUM_WORKERS = 4
IMAGE_SIZE = 224
CROP_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Data structure for image metadata
@dataclass
class ImageMetadata:
    image_path: str
    label: int

# Enum for data split
class DataSplit(Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'

# Abstract base class for dataset
class DatasetBase(ABC):
    def __init__(self, data_dir: str, split: DataSplit):
        self.data_dir = data_dir
        self.split = split
        self.metadata = self.load_metadata()

    @abstractmethod
    def load_metadata(self) -> List[ImageMetadata]:
        pass

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        metadata = self.metadata[index]
        image_path = metadata.image_path
        label = metadata.label

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess image
        image = self.preprocess_image(image)

        return image, label

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image / 255.0
        image = (image - MEAN) / STD
        return image

# Dataset class for training data
class TrainingDataset(DatasetBase):
    def load_metadata(self) -> List[ImageMetadata]:
        metadata_path = os.path.join(self.data_dir, f'{self.split.value}_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return [ImageMetadata(image_path, label) for image_path, label in metadata.items()]

# Dataset class for validation and test data
class ValidationTestDataset(DatasetBase):
    def load_metadata(self) -> List[ImageMetadata]:
        metadata_path = os.path.join(self.data_dir, f'{self.split.value}_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return [ImageMetadata(image_path, label) for image_path, label in metadata.items()]

# Data loader class
class DataLoaderClass:
    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_loader = DataLoader(dataset, batch_size, num_workers)

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

# Create data loader
def create_data_loader(data_dir: str, split: DataSplit, batch_size: int, num_workers: int) -> DataLoaderClass:
    if split == DataSplit.TRAIN:
        dataset = TrainingDataset(data_dir, split)
    else:
        dataset = ValidationTestDataset(data_dir, split)
    data_loader = DataLoaderClass(dataset, batch_size, num_workers)
    return data_loader

# Main function
def main():
    data_dir = DATA_DIR
    split = DataSplit.TRAIN
    batch_size = BATCH_SIZE
    num_workers = NUM_WORKERS

    data_loader = create_data_loader(data_dir, split, batch_size, num_workers)
    logger.info(f'Created data loader for {split.value} split')

    for batch in data_loader:
        images, labels = batch
        logger.info(f'Batch size: {len(images)}')
        logger.info(f'Images shape: {images.shape}')
        logger.info(f'Labels shape: {labels.shape}')

if __name__ == '__main__':
    main()