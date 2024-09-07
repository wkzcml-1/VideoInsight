import cv2
import hashlib
import logging
import os

logger = logging.getLogger(__name__)

class VideoProcessor:
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
            print("Error: Could not open video file.")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = cap.get(cv2.CAP_PROP_FOURCC) 
        duration = frame_count / fps if fps > 0 else None

        cap.release()
        
        return {
            'file_path': video_path,
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'codec': codec,
            'duration': duration
        }

