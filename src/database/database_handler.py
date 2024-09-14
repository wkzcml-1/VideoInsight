import os
import logging

from milvus_handler import MilvusHandler
from mongodb_handler import MongoDBHandler

from collections import defaultdict

from utils.load_config import load_config

logger = logging.getLogger(__name__)


# unified interface for mongodb and milvus
class DatabaseHandler:
    # initialize the database handler
    def __init__(self):
        self.milvus_handler = MilvusHandler()
        self.mongodb_handler = MongoDBHandler()

    # check if the video is already registered
    def check_video_registered(self, video_hash):
        return self.mongodb_handler.check_video_registered(video_hash)
    
    # generate a video id
    def generate_video_id(self, video_hash):
        result = self.check_video_registered(video_hash)
        if result:
            return result['video_id']
        else:
            return self.mongodb_handler.generate_video_id()
    
    # insert video metadata
    def insert_video_metadata(self, video_id, metadata):
        try:
            self.mongodb_handler.insert_video_metadata(video_id, metadata)
        except Exception as e:
            logger.error(f"Error inserting video metadata: {e}")
    
    # insert visual segment
    def insert_visual_segment(self, video_id, visual_info):
        try:
            segment, entity = {}, {}
            # split visual_info into segment and entity according to keys containing 'vector'
            for key in list(visual_info.keys()):
                if 'vector' in key:
                    entity[key] = visual_info.pop(key)
                else:
                    segment[key] = visual_info.pop(key)
            # insert in mongodb
            segment_id = self.mongodb_handler.insert_visual_segment(video_id, segment)
            # insert in milvus
            entity['video_id'] = video_id
            entity['segment_id'] = segment_id
            self.milvus_handler.insert_visual_segment(entity)
        except Exception as e:
            logger.error(f"Error inserting visual segment: {e}")

    # insert audio segment
    def insert_audio_segment(self, video_id, audio_info):
        try:
            segment, entity = {}, {}
            # split audio_info into segment and entity according to keys containing 'vector'
            for key in list(audio_info.keys()):
                if 'vector' in key:
                    entity[key] = audio_info.pop(key)
                else:
                    segment[key] = audio_info.pop(key)
            # insert in mongodb
            segment_id = self.mongodb_handler.insert_audio_segment(video_id, segment)
            # insert in milvus
            entity['video_id'] = video_id
            entity['segment_id'] = segment_id
            self.milvus_handler.insert_audio_segment(entity)
        except Exception as e:
            logger.error(f"Error inserting audio segment: {e}")

    # search by time range
    def search_by_time_range(self, video_id, start_time, end_time, collection):
        try:
            if collection == 'visual':
                return self.mongodb_handler.search_visual_segments_by_time(video_id, start_time, end_time)
            elif collection == 'audio':
                return self.mongodb_handler.search_audio_segments_by_time(video_id, start_time, end_time)
            else:
                raise ValueError("Invalid collection name in search_by_time_range")
        except Exception as e:
            logger.error(f"Error searching by time range: {e}")

    # search by semantic vector
    def search_by_query_vectors(self, video_id, vectors, collection, topk=10):
        try:
            if collection == 'visual':
                hits = self.milvus_handler.search_visual_vectors(video_id, vectors, topk)
                # search hits in mongodb，hits: dict (video_id: [segment_id...])
                ret = {}
                for video_id, segment_ids in hits.items():
                    ret[video_id] = self.mongodb_handler.get_visual_segments(video_id, segment_ids)
                return ret
            elif collection == 'audio':
                hits = self.milvus_handler.search_audio_vectors(video_id, vectors, topk)
                # search hits in mongodb，hits: dict (video_id: [segment_id...])
                ret = {}
                for video_id, segment_ids in hits.items():
                    ret[video_id] = self.mongodb_handler.get_audio_segments(video_id, segment_ids)
                return ret
            else:
                raise ValueError("Invalid collection name in search_by_query_vectors")
        except Exception as e:
            logger.error(f"Error searching by query vector: {e}")
    

    # delete video_info by video_id
    def delete_by_video_id(self, video_id):
        try:
            self.mongodb_handler.delete_by_video_id(video_id)
            self.milvus_handler.delete_by_video_id(video_id)
        except Exception as e:
            logger.error(f"Error deleting video info: {e}")
    
    # drop all collections
    def drop_all_collections(self):
        try:
            self.mongodb_handler.drop_all_collections()
            self.milvus_handler.drop_all_collections()
        except Exception as e:
            logger.error(f"Error dropping collections: {e}")
    
        

        


