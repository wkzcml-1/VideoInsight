import cv2
import hashlib
import logging
import os

from scene_segmentation import scene_segmentation
from scene_segmentation import extract_frames
from scene_segmentation import extract_key_frames
from scene_segmentation import sample_frames

from models.ASRModel import WhisperModel
from models.ImgTxtRetriModel import BAAIAltCLIPModel
from models.TxtEmbModel import BgeM3Model
from models.RerankerModel import BgeRerankerModel

from database.database_handler import DatabaseHandler

from utils.load_config import load_config

config = load_config()

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, mllm, **kwargs):
        self.database_handler = DatabaseHandler()
        # load mllm model
        self.mllm = mllm
        # load ASR model
        self.asr_model = WhisperModel(
            model_id=config['ASR_MODEL']['MODEL_ID'],
            device=config['ASR_MODEL']['DEVICE']
        )
        # load text embedding model
        self.text_emb_model = BgeM3Model(
            model_id=config['TEXT_EMBEDDING_MODEL']['MODEL_ID'],
            device=config['TEXT_EMBEDDING_MODEL']['DEVICE']
        )
        # load image text retrieval model
        self.img_txt_retri_model = BAAIAltCLIPModel(
            model_id=config['IMAGE_TEXT_RETRIEVAL_MODEL']['MODEL_ID'],
            device=config['IMAGE_TEXT_RETRIEVAL_MODEL']['DEVICE']
        )
        # load reranker model
        self.reranker_model = BgeRerankerModel(
            model_id=config['RERANKER_MODEL']['MODEL_ID'],
            device=config['RERANKER_MODEL']['DEVICE']
        )

    def generate_video_hash(self, video_path, algorithm='sha256', block_size=65536):
        h = hashlib.new(algorithm)
        # read the video file in chunks and update the hash
        with open(video_path, 'rb') as f:
            while chunk := f.read(block_size):
                h.update(chunk)
        return h.hexdigest()
    
    def get_video_metadata(video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Error opening video: {video_path}")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = cap.get(cv2.CAP_PROP_FOURCC) 
        duration = frame_count / fps if fps > 0 else None

        cap.release()
        
        return {
            'path': video_path,
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'codec': codec,
            'duration': duration
        }
    
    def check_video_registered(self, video_path):
        video_hash = self.generate_video_hash(video_path)
        return self.database_handler.check_video_registered(video_hash)
    
    # encode text into dense vector, sparse vector, and encode image into clip vector
    def encode_segment(self, text, image=None):
        # encode text into dense vector and sparse vector
        dense_vector, sparse_vector = self.text_emb_model.encode(text)
        # if image is provided, encode image into clip vector
        clip_vector = None
        if image:
            clip_vector = self.img_txt_retri_model.image_encode(image)
        if clip_vector is None:
            return {'dense_vector': dense_vector, 'sparse_vector': sparse_vector}
        return {'dense_vector': dense_vector, 'sparse_vector': sparse_vector, 'clip_vector': clip_vector}
    

    def process_video(self, video_path):
        # check if video is already registered
        video_metadata = self.check_video_registered(video_path)
        if video_metadata:
            return video_metadata['video_id']
        
        # get video metadata
        video_metadata = self.get_video_metadata(video_path)
        if not video_metadata:
            return None
        # generate video hash
        video_hash = self.generate_video_hash(video_path)
        # insert video_hash into video metadata
        video_metadata['hash'] = video_hash
        # generate video id
        video_id = self.database_handler.generate_video_id(video_hash)
        # insert video metadata
        self.database_handler.insert_video_metadata(video_id, video_metadata)

        # process audio information
        audio_segments = self.asr_model.transcribe(video_path)
        for audio_segment in audio_segments:
            transcript = audio_segment['text']
            start_time, end_time = audio_segment['timestamp']
            # encode audio segment
            audio_info = self.encode_segment(transcript)
            audio_info['start_time'] = start_time
            audio_info['end_time'] = end_time
            audio_info['transcript'] = transcript
            # insert audio segment
            self.database_handler.insert_audio_segment(video_id, audio_info)
        # unload ASR model
        self.asr_model.unload_model()

        # process visual information
        scene_list = scene_segmentation(video_path)
        for scene in scene_list:
            start_time, end_time = scene[0].get_seconds(), scene[1].get_seconds()
            start_frame, end_frame = scene[0].get_frames(), scene[1].get_frames()
            # extract frames from scene
            frames = extract_frames(video_path, start_frame, end_frame)
            key_frame = extract_key_frames(frames)
            # sample frames from scene
            target_frame_count = config['VIDEO_PROCESSING']['TARGET_FRAME_COUNT']
            sampled_frames = sample_frames(frames, target_frame_count=target_frame_count)
            # get description of scene
            prompt

        
        return video_id

