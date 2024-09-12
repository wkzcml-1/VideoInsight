from abc import ABC, abstractmethod

import os
import logging

from pymilvus.model.reranker import BGERerankFunction
from utils.project_paths import CHECKPOINTS_DIR
from utils.clear_memory import clear_memory

logger = logging.getLogger(__name__)

class RerankerModel(ABC):
    @abstractmethod
    def rerank(self, query, documents, top_k=10):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def unload_model(self):
        pass


# BAAI/bge-reranker-v2-m3
class BgeRerankerModel(RerankerModel):
    def __init__(self, **kwargs):
        path = kwargs.get('model_id', 'BAAI/bge-reranker-v2-m3')
        self.model_id = os.path.join(CHECKPOINTS_DIR, path)
        self.device = kwargs.get('device', 'cpu')
        self.load_model()

    def load_model(self):
        use_fp16 = True
        if self.device == 'cpu':
            use_fp16 = False
        self.model = BGERerankFunction(
            model_name=self.model_id,
            device=self.device,
            use_fp16=use_fp16
        )
        logger.info(f"BGE Reranker model loaded from {self.model_id} on {self.device}")
    
    def unload_model(self):
        if self.model is None:
            return
        del self.model.reranker
        del self.model
        clear_memory()
        self.model = None
    
    def rerank(self, query, documents, top_k=10):
        if self.model is None:
            self.load_model()
        results = self.model(query, documents, top_k)
        return [res.index for res in results]
    
    def __del__(self):
        self.unload_model()
        clear_memory()
        logger.info("BGE Reranker model unloaded")