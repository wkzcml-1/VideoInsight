import numpy as np
from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import os, logging
import librosa, datetime
from utils.project_paths import CHECKPOINTS_DIR
from utils.clear_memory import clear_memory

import whisper

logger = logging.getLogger(__name__)

class ASRModel(ABC):
    @abstractmethod
    def transcribe(self, audio, sample_rate=16_000):
        pass
    
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def unload_model(self):
        pass


class WhisperModel(ASRModel):
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cpu')
        path = kwargs.get('model_id', 'openai/whisper-large-v3')
        self.model_id = os.path.join(CHECKPOINTS_DIR, path, 'model.pt')
        self.model = None
        if kwargs.get('load', True):
            self.load_model()
        
    def load_model(self):
        if self.model is not None:
            return
        self.model = whisper.load_model(self.model_id, device=self.device)
        logger.info(f"Whisper model loaded from {self.model_id}")
        clear_memory()
    
    def unload_model(self):
        if self.model is None:
            return
        del self.model
        clear_memory()
        self.model = None
    
    def transcribe(self, audio_path, min_duration=8, eps=0.5):
        if self.model is None:
            self.load_model()
        segments = self.model.transcribe(
            audio_path,
            temperature=0.0,
            compression_ratio_threshold=1.8,
            condition_on_previous_text=False
        )
        results =  [{'timestamp': (seg['start'], seg['end']), 'text': seg['text']} for seg in segments['segments']]
        # merge short segments to avoid too many short segments
        if len(results) > 1:
            merged_results = []
            curr_seg = None
            for seg in results:
                if curr_seg is None:
                    curr_seg = seg
                else:
                    if abs(seg['timestamp'][0] - curr_seg['timestamp'][1]) < eps:
                        curr_seg['timestamp'] = (curr_seg['timestamp'][0], seg['timestamp'][1])
                        curr_seg['text'] += ' ' + seg['text']
                    else:
                        # update the current segment
                        merged_results.append(curr_seg)
                        curr_seg = seg
                # judge if the current segment is long enough
                if curr_seg['timestamp'][1] - curr_seg['timestamp'][0] >= min_duration:
                    merged_results.append(curr_seg)
                    curr_seg = None
            
            if curr_seg is not None:
                merged_results.append(curr_seg)
            results = merged_results
        return results
        

# whisper model hugingface implementation
class WhisperModel_huggingface(ASRModel):
    def __init__(self, **kwargs):
        path = kwargs.get('model_id', 'openai/whisper-large-v3')
        self.model_id = os.path.join(CHECKPOINTS_DIR, path)
        self.device = kwargs.get('device', 'cpu')

        self.torch_dtype = torch.float32 if self.device == 'cpu' else torch.float16
        self.model = None
        if kwargs.get('load', True):
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
            chunk_length_s=30,
        )
        result = pipe(audio, return_timestamps=True)['chunks']
            
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


        