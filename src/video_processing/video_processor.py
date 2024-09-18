import cv2
import hashlib
import logging
import os

from .scene_segmentation import scene_segmentation
from .scene_segmentation import extract_frames
from .scene_segmentation import extract_key_frames
from .scene_segmentation import sample_frames

from models.ASRModel import WhisperModel
from models.ImgTxtRetriModel import BAAIAltCLIPModel
from models.TxtEmbModel import BgeM3Model
from models.RerankerModel import BgeRerankerModel

from database.database_handler import DatabaseHandler

from utils.load_config import load_config
from utils.clear_memory import clear_memory
from utils.prompts import VIDEO_DESCRIPTION_PROMPT

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
            device=config['ASR_MODEL']['DEVICE'],
            load=False
        )
        # load text embedding model
        self.text_emb_model = BgeM3Model(
            model_id=config['TEXT_EMBEDDING_MODEL']['MODEL_ID'],
            device=config['TEXT_EMBEDDING_MODEL']['DEVICE'],
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
    
    @staticmethod
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
            'name': os.path.basename(video_path),
            'path': video_path,
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'codec': codec,
            'duration': duration
        }
    
    def get_all_video_metadata(self):
        return self.database_handler.get_all_video_metadata()

    def check_video_registered(self, video_path):
        video_hash = self.generate_video_hash(video_path)
        return self.database_handler.check_video_registered(video_hash)

    def get_video_summary(self, video_id):
        return self.database_handler.get_video_summary(video_id)
    
    # encode text into dense vector, sparse vector, and encode image into clip vector
    def encode_segment(self, text, image=None, generate_clip_vector=False):
        # encode text into dense vector and sparse vector
        dense_vector, sparse_vector = self.text_emb_model.encode(text)
        dense_vector, sparse_vector = dense_vector[0], sparse_vector[0]
        # if image is provided, encode image into clip vector
        clip_vector = None
        if image and generate_clip_vector:
            clip_vector = self.img_txt_retri_model.image_encode(image)
            clip_vector = clip_vector[0]
        elif generate_clip_vector:
            clip_vector = self.img_txt_retri_model.text_encode(text)
            clip_vector = clip_vector[0]
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
        # unload mllm model
        self.mllm.unload_model()
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
        scene_list = scene_segmentation(video_path, config['VIDEO_PROCESSING']['MIN_SCENE_DURATION'])
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
            prompt = self.mllm.create_prompt(VIDEO_DESCRIPTION_PROMPT, frames=sampled_frames)
            description = self.mllm.generate(prompt)
            # encode description and key frame
            visual_info = self.encode_segment(description, image=key_frame, generate_clip_vector=True)
            visual_info['start_time'] = start_time
            visual_info['end_time'] = end_time
            visual_info['description'] = description
            # insert visual segment
            self.database_handler.insert_visual_segment(video_id, visual_info)
        # unload MLLM model
        self.mllm.unload_model()
        return video_id
    
    def insert_summary(self, video_id, scene_id, summary):
        self.database_handler.insert_summary(video_id, scene_id, summary)

    def get_num_of_visual_segments(self, video_id):
        return self.database_handler.get_num_of_visual_segments(video_id)
    
    def get_num_of_audio_segments(self, video_id):
        return self.database_handler.get_num_of_audio_segments(video_id)
    
    def get_visual_segments(self, video_id, segment_id=None):
        return self.database_handler.get_visual_segments(video_id, segment_id)
    
    def get_audio_segments(self, video_id, segment_id=None):
        return self.database_handler.get_audio_segments(video_id, segment_id)
    
    def extract_key_frame(self, video_id, start_time, end_time):
        video_metadata = self.database_handler.search_video_metadata(video_id)
        video_path, fps = video_metadata['path'], video_metadata['fps']
        start_frame, end_frame = int(start_time * fps), int(end_time * fps)
        frames = extract_frames(video_path, start_frame, end_frame)
        key_frame = extract_key_frames(frames)
        return key_frame

    def search_video_by_time(self, video_id, start_time, end_time=None, collection='both'):
        visual_segments, audio_segments = [], []
        if collection == 'visual' or collection == 'both':
            visual_segments = self.database_handler.search_by_time_range('visual', video_id, start_time, end_time)
            visual_segments = list(visual_segments)
        if collection == 'audio' or collection == 'both':
            audio_segments = self.database_handler.search_by_time_range('audio', video_id, start_time, end_time)
            audio_segments = list(audio_segments)
        if collection == 'both':
            return visual_segments, audio_segments
        return visual_segments if collection == 'visual' else audio_segments
    
    def search_video_by_semantic(self, video_id, query, **kwargs):
        # parameters
        v_top_k = kwargs.get('v_top_k', 5)
        a_top_k = kwargs.get('a_top_k', 5)
        # encode query
        query_vector = self.encode_segment(query, generate_clip_vector=True)
        # hybrid search in milvus
        # search visual segments
        hits = self.database_handler.search_by_query_vectors('visual', video_id, query_vector, topk=v_top_k)
        visual_recalls = list(hits[video_id])
        # rerank visual segments
        documents = [segment['description'] for segment in visual_recalls]
        visual_indices = self.reranker_model.rerank(query, documents, top_k=v_top_k)
        # filter visual segments from visual recalls by reranked indices
        visual_segments = [visual_recalls[i] for i in visual_indices]
        # search audio segments
        hits = self.database_handler.search_by_query_vectors('audio', video_id, query_vector, topk=a_top_k)
        audio_recalls = list(hits[video_id])
        # rerank audio segments
        documents = [segment['transcript'] for segment in audio_recalls]
        audio_indices = self.reranker_model.rerank(query, documents, top_k=a_top_k)
        # filter audio segments from audio recalls by reranked indices
        audio_segments = [audio_recalls[i] for i in audio_indices]

        return visual_segments, audio_segments
    
    def search_video_metadata(self, video_id):
        return self.database_handler.search_video_metadata(video_id)
        
    def delete_by_video_id(self, video_id):
        self.database_handler.delete_by_video_id(video_id)

    def drop_all(self):
        self.database_handler.drop_all_collections()
        
