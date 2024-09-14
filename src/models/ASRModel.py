import numpy as np
from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import os, logging
import librosa, datetime
from utils.project_paths import CHECKPOINTS_DIR
from utils.clear_memory import clear_memory
from utils.project_paths import DEBUG_DIR

logger = logging.getLogger(__name__)

class ASRModel(ABC):
    @abstractmethod
    def transcribe(self, audio, sample_rate=16_000):
        pass

    @abstractmethod
    def transcribe_file(self, audio_path):
        pass
    
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def unload_model(self):
        pass

    

# whisper model
class WhisperModel(ASRModel):
    def __init__(self, **kwargs):
        path = kwargs.get('model_id', 'openai/whisper-large-v3')
        self.model_id = os.path.join(CHECKPOINTS_DIR, path)
        self.device = kwargs.get('device', 'cpu')
        self.debug = kwargs.get('debug', False)

        self.torch_dtype = torch.float32 if self.device == 'cpu' else torch.float16
        self.model = None
        self.load_model()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
       
    def transcribe(self, audio, sample_rate=16_000):
        if self.model is None:
            self.load_model()
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        result = pipe(audio, return_timestamps=True)['chunks']

        if self.debug:
            ASR_DEBUG_DIR = os.path.join(DEBUG_DIR, 'asr')
            os.makedirs(ASR_DEBUG_DIR, exist_ok=True)
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            debug_path = os.path.join(ASR_DEBUG_DIR, f'whisper_{now}.txt')
            with open(debug_path, 'w') as f:
                for segment in result:
                    f.write(f"[{segment['timestamp'][0]} -> {segment['timestamp'][1]}] {segment['text']}\n")
            logger.info(f"Debug results saved to {debug_path}")
            
        return result
    
    def transcribe_file(self, audio_path):
        audio, fs = librosa.load(audio_path, sr=16_000)
        logger.info(f"Transcribing {audio_path} with sample rate {fs}")
        return self.transcribe(audio, sample_rate=fs)
    
    def load_model(self):
        if self.model is not None:
            return
        # if self.device == 'cuda', use flash-attn
        if 'cuda' in self.device:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2"
            )
        else:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True
            )
        self.model.to(self.device)
        logger.info(f"Whisper model loaded from {self.model_id}")
        clear_memory()
        
    def unload_model(self):
        if self.model is None:
            return
        del self.model
        clear_memory()
        self.model = None


        