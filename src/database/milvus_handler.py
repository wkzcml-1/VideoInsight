import os
import logging
import pymilvus

from collections import defaultdict

from pymilvus import (
    connections, utility, DataType,
    FieldSchema, CollectionSchema, 
    Collection, AnnSearchRequest,
)

logger = logging.getLogger(__name__)

# Get the database connection VARCHAR from the environment
milvus_port = os.getenv('MILVUS_LOCAL_PORT', 19530)

class MilvusHandler:
    def __init__(self, host='localhost', port=milvus_port):
        connections.connect("default", host=host, port=port)
        self.setup_collections()
        self.visual_collection = Collection(name='visual_segments')
        self.audio_collection = Collection(name='audio_segments')

    def setup_collections(self):
        # index parameters
        sparse_index_params = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        dense_index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}

        # create visual_segments collection if it doesn't exist
        if not utility.has_collection('visual_segments'):
            # schema of the collection: video_id, segment_id, dense_vector, sparse_vector, clip_vector
            visual_segments_schema = [
                FieldSchema(name='video_segment_id', dtype=DataType.VARCHAR, max_length=50, is_primary=True),
                FieldSchema(name='video_id', max_length=40, dtype=DataType.VARCHAR),
                FieldSchema(name='segment_id', dtype=DataType.INT32),
                FieldSchema(name='dense_vector', dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name='sparse_vector', dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name='clip_vector', dtype=DataType.FLOAT_VECTOR, dim=768)
            ]
            collection_schema = CollectionSchema(fields=visual_segments_schema, description="Visual Segments")
            col = Collection(name='visual_segments', schema=collection_schema)
            # create index for visual_segments collection
            col.create_index(field_name='dense_vector', index_params=dense_index_params)
            col.create_index(field_name='sparse_vector', index_params=sparse_index_params)
            col.create_index(field_name='clip_vector', index_params=dense_index_params)
            logger.info("Collection visual_segments created")
        else:
            logger.info("Collection visual_segments already exists")
        
        # create audio_segments collection if it doesn't exist
        if not utility.has_collection('audio_segments'):
            # schema of the collection: video_id, segment_id, dense_vector, sparse_vector
            audio_segments_schema = [
                FieldSchema(name='video_segment_id', dtype=DataType.VARCHAR, max_length=50, is_primary=True),
                FieldSchema(name='video_id', max_length=40, dtype=DataType.VARCHAR),
                FieldSchema(name='segment_id', dtype=DataType.INT32),
                FieldSchema(name='dense_vector', dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name='sparse_vector', dtype=DataType.SPARSE_FLOAT_VECTOR)
            ]
            collection_schema = CollectionSchema(fields=audio_segments_schema, description="Audio Segments")
            col = Collection(name='audio_segments', schema=collection_schema)
            # create index for audio_segments collection
            col.create_index(field_name='dense_vector', index_params=dense_index_params)
            col.create_index(field_name='sparse_vector', index_params=sparse_index_params)
            logger.info("Collection audio_segments created")
        else:
            logger.info("Collection audio_segments already exists")

    @staticmethod
    def check_entity(entity, collection_name):
        assert 'video_id' in entity, "video_id key missing in entity"
        assert 'segment_id' in entity, "segment_id key missing in entity"
        assert 'dense_vector' in entity, "dense_vector key missing in entity"
        assert 'sparse_vector' in entity, "sparse_vector key missing in entity"
        if collection_name == 'visual_segments':
            assert 'clip_vector' in entity, "clip_vector key missing in entity"
        elif collection_name != 'audio_segments':
            raise ValueError("Invalid collection name")

    def insert_visual_segment(self, entity):
        # check collection exists
        if not utility.has_collection('visual_segments'):
            # create collection if it doesn't exist
            self.setup_collections()
        # entity can be a dictionary with keys or a list of dictionaries
        if isinstance(entity, list):
            for e in entity:
                self.check_entity(e, 'visual_segments')
        elif isinstance(entity, dict):
            self.check_entity(entity, 'visual_segments')
            entity = [entity]
        # delete other keys except the ones mentioned above
        keys_to_keep = ['video_id', 'segment_id', 'dense_vector', 'sparse_vector', 'clip_vector']
        for idx, e in enumerate(entity):
            entity[idx] = {k: e[k] for k in keys_to_keep}
            # create video_segment_id
            entity[idx]['video_segment_id'] = f"{e['video_id']}@{e['segment_id']}"
        self.visual_collection.insert(entity)
        # logging
        video_ids = [e['video_id'] for e in entity]
        segment_ids = [e['segment_id'] for e in entity]
        logger.info(f"Visual segment inserted for video_ids: {video_ids}, segment_ids: {segment_ids}")

    def insert_audio_segment(self, entity):
        # check collection exists
        if not utility.has_collection('audio_segments'):
            # create collection if it doesn't exist
            self.setup_collections()
        # entity can be a dictionary with keys or a list of dictionaries
        if isinstance(entity, list):
            for e in entity:
                self.check_entity(e, 'audio_segments')
        elif isinstance(entity, dict):
            self.check_entity(entity, 'audio_segments')
            entity = [entity]
        # delete other keys except the ones mentioned above
        keys_to_keep = ['video_id', 'segment_id', 'dense_vector', 'sparse_vector']
        for idx, e in enumerate(entity):
            entity[idx] = {k: e[k] for k in keys_to_keep}
            # create video_segment_id
            entity[idx]['video_segment_id'] = f"{e['video_id']}@{e['segment_id']}"
        self.audio_collection.insert(entity)
        # logging
        video_ids = [e['video_id'] for e in entity]
        segment_ids = [e['segment_id'] for e in entity]
        logger.info(f"Audio segment inserted for video_ids: {video_ids}, segment_ids: {segment_ids}")
    
    @staticmethod
    def search_vector(col, video_id, vector_name, vector, top_k=10):
        expr = f"video_id == '{video_id}'"
        hits = col.search(
            [vector],
            anns_field=vector_name,
            limit=top_k,
            output_fields=['video_id', 'segment_id'],
            param={"metric_type": "IP", "params": {}},
            expr=expr
        )[0]
        return [(hit.entity.get('video_id'), hit.entity.get('segment_id')) for hit in hits]
    
    def search_visual_vectors(self, video_id, vectors, top_k=10):
        self.visual_collection.load()
        # vectors must be a dictionary with keys in
        # [dense_vector, sparse_vector, clip_vector] 
        vector_names = ['dense_vector', 'sparse_vector', 'clip_vector']
        hits = []
        for vector_name, vector in vectors.items():
            if vector_name not in vector_names:
                continue
            hits += self.search_vector(self.visual_collection, video_id, vector_name, vector, top_k)
            logger.info(f"Search results for {vector_name}: {hits}")
        # remove duplicates
        hits_rd = list(set(hits))
        self.visual_collection.release()
        # combine same video_id into dictionary
        hits_dict = defaultdict(list)
        for hit in hits_rd:
            hits_dict[hit[0]].append(hit[1])    
        return hits_dict
    
    def search_audio_vectors(self, video_id, vectors, top_k=10):
        self.audio_collection.load()
        # vectors must be a dictionary with keys in
        # [dense_vector, sparse_vector] 
        vector_names = ['dense_vector', 'sparse_vector']
        hits = []
        for vector_name, vector in vectors.items():
            if vector_name not in vector_names:
                continue
            hits += self.search_vector(self.audio_collection, video_id, vector_name, vector, top_k)
            logger.info(f"Search results for {vector_name}: {hits}")
        # remove duplicates
        hits_rd = list(set(hits))
        self.audio_collection.release()
        # combine same video_id into dictionary
        hits_dict = defaultdict(list)
        for hit in hits_rd:
            hits_dict[hit[0]].append(hit[1])
        return hits_dict
    
    def delete_by_video_id(self, video_id):
        # load collections
        self.visual_collection.load()
        self.audio_collection.load()
        self.visual_collection.delete(expr=f"video_id == '{video_id}'")
        self.audio_collection.delete(expr=f"video_id == '{video_id}'")
        # release collections
        self.visual_collection.release()
        self.audio_collection.release()
        logger.info(f"Deleted all segments for video_id: {video_id}")
    
    def drop_all_collections(self):
        try:
            self.visual_collection.drop()
            self.audio_collection.drop()
            logger.info("All milvus collections dropped")
        except Exception as e:
            logger.error(f"Error dropping collections: {e}")
