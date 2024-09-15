from abc import ABC, abstractmethod

from PIL import Image
from transformers import AltCLIPModel, AltCLIPProcessor

import numpy as np
import os
import logging
import torch

from utils.project_paths import CHECKPOINTS_DIR
from utils.clear_memory import clear_memory

logger = logging.getLogger(__name__)

class ImgTxtRetriModel(ABC):
    @abstractmethod
    def image_encode(self, image):
        pass

    def text_encode(self, text):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def unload_model(self):
        pass


# BAAI AltCLIP model
class BAAIAltCLIPModel(ImgTxtRetriModel):
    def __init__(self, **kwargs):
        path = kwargs.get('model_id', 'BAAI/AltCLIP')
        self.model_id = os.path.join(CHECKPOINTS_DIR, path)
        self.device = kwargs.get('device', 'cpu')
        if kwargs.get('load', True):
            self.load_model()

    def load_model(self):
        torch_dtype = torch.float32
        if 'cuda' in self.device:
            torch_dtype = torch.float16
            
        self.model = AltCLIPModel.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=torch_dtype
        )
        self.model.eval()

        self.processor = AltCLIPProcessor.from_pretrained(self.model_id)
        logger.info(f"AltCLIP model loaded from {self.model_id} on {self.device}")

    def unload_model(self):
        if self.model is None:
            return
        del self.model
        clear_memory()
        self.model = None

    def image_encode(self, image):
        if self.model is None:
            self.load_model()
        if isinstance(image, str):
            image = Image.open(image)
        inputs = self.processor(
            images=image, return_tensors="pt"
        )
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.numpy()

    def text_encode(self, text):
        if self.model is None:
            self.load_model()
        inputs = self.processor(
            text=text, return_tensors="pt", padding=True
        )
        scale = self.model.logit_scale.exp().item()
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features * scale
        return text_features.numpy()