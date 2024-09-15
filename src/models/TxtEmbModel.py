from abc import ABC, abstractmethod

import os
import logging

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from utils.project_paths import CHECKPOINTS_DIR
from utils.clear_memory import clear_memory

logger = logging.getLogger(__name__)

class TxtEmbModel(ABC):
    @abstractmethod
    def encode(self, text):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def unload_model(self):
        pass


# bge m3 model
class BgeM3Model(TxtEmbModel):
    def __init__(self, **kwargs):
        path = kwargs.get('model_id', 'BAAI/bge-m3')
        self.model_id = os.path.join(CHECKPOINTS_DIR, path)
        self.device = kwargs.get('device', 'cpu')
        if kwargs.get('load', True):
            self.load_model()

    def load_model(self):
        use_fp16 = True
        if self.device == 'cpu':
            use_fp16 = False
        self.model = BGEM3EmbeddingFunction(
            model_name=self.model_id,
            device=self.device,
            use_fp16=use_fp16
        )
        logger.info(f"BGE M3 model loaded from {self.model_id} on {self.device}")
    
    def unload_model(self):
        if self.model is None:
            return
        del self.model.model
        del self.model
        clear_memory()
        self.model = None
    
    def encode(self, texts):
        if self.model is None:
            self.load_model()
        if isinstance(texts, str):
            texts = [texts]
        ret = self.model(texts)
        dense, sparse = ret['dense'], ret['sparse']
        # split sparse into list
        sparse = [sparse[[i], :] for i in range(sparse.shape[0])]
        return dense, sparse
    
    def __del__(self):
        self.unload_model()
        logger.info(f"BGE M3 model unloaded")