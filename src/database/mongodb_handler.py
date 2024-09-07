import logging
from pymongo import MongoClient, ASCENDING

import uuid
import os
from urllib.parse import quote

import hashlib

logger = logging.getLogger(__name__)

# Get the database connection string from the environment
mongodb_user = os.getenv('MONGO_INITDB_ROOT_USERNAME')
mongodb_password = quote(os.getenv('MONGO_INITDB_ROOT_PASSWORD'))
mongodb_port = os.getenv('MONGO_DB_LOCAL_PORT')

class MongoDBHandler:
    def __init__(self, uri=None, db_name='video_insight_database'):
        if uri is None:
            uri = f'mongodb://{mongodb_user}:{mongodb_password}@localhost:{mongodb_port}'
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.setup_indexes()
    
    def setup_indexes(self):
        # 为 video_segments 和 asr_data 建立索引
        self.db.videos_metadata.create_index([("video_id", ASCENDING)], unique=True)
        self.db.video_segments.create_index([("video_id", ASCENDING), ("start_time", ASCENDING), ("end_time", ASCENDING)], unique=True)
        self.db.video_transcripts.create_index([("video_id", ASCENDING), ("start_time", ASCENDING), ("end_time", ASCENDING)], unique=True)

    def generate_video_id(self):
        while True:
            video_id = str(uuid.uuid4())
            if not self.db.videos_metadata.find_one({'video_id': video_id}):
                return video_id
        return None
    
    def check_video_registered(self, video_hash):
        return self.db.videos_metadata.find_one({'hash': video_hash})

    def insert_video_metadata(self, video_id, metadata):
        assert 'path' in metadata, "Metadata should have path"
        assert 'hash' in metadata, "Metadata should have hash"
        # check if hash is provided
        if self.check_video_registered(metadata['hash']):
            logger.info(f"Video already registered with hash: {metadata['hash']}")
            return
        # combine video_id with metadata
        metadata['video_id'] = video_id
        self.db.videos_metadata.insert_one(metadata)
        logger.info(f"Video metadata inserted for video_id: {video_id}")
    
    def insert_video_segment(self, video_id, segment):
        # check
        assert 'start_time' in segment and 'end_time' in segment,  \
            "Segment should have start_time and end_time"   
        
        segment['video_id'] = video_id
        self.db.video_segments.insert_one(segment)
        logger.info(f"Video segment inserted for video_id: {video_id}")

    def insert_video_transcript(self, video_id, transcript):
        # check
        assert 'start_time' in transcript and 'end_time' in transcript, \
            "Transcript should have start_time and end_time"
        
        transcript['video_id'] = video_id
        self.db.video_transcripts.insert_one(transcript)
        logger.info(f"Video transcript inserted for video_id: {video_id}")
        
    def delete_video_info(self, video_id):
        self.db.videos_metadata.delete_one({'video_id': video_id})
        logger.info(f"Video metadata deleted for video_id: {video_id}")
        # delete all segments and transcripts
        self.db.video_segments.delete_many({'video_id': video_id})
        self.db.video_transcripts.delete_many({'video_id': video_id})
        logger.info(f"Video segments and transcripts deleted for video_id: {video_id}")
        
        
