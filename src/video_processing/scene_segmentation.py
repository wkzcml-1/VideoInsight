import cv2
import os
import time
import logging

import numpy as np

from utils.clear_memory import clear_memory
from utils.project_paths import DEBUG_DIR

from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg

logger = logging.getLogger(__name__)

def scene_segmentation(video_path, debug=False):
    """
    Detects scene changes in a video using scenedetect library.
    Parameters
    ----------
    video_path : str
        Path to the video file.
    debug : bool, optional
        If True, saves the segmented scenes to the output directory. The default is False.
    Returns
    -------
    List of tuples
        List of tuples containing the start and end frame numbers of each scene.
    """
    # clear memory
    clear_memory()

    # record processing time
    start_time = time.time()

    # scenedetect
    detector = AdaptiveDetector()
    scene_list = detect(video_path, detector)

    # log processing time
    processing_time = time.time() - start_time
    logger.info(f"Scene segmentation for {video_path} completed in {processing_time:.2f} seconds")

    # print scene list
    if debug:
        # create video_processing in debug directory
        debug_dir = os.path.join(DEBUG_DIR, 'video_processing')
        # create subdirectory with video name and processing time
        video_name = os.path.basename(video_path).split('.')[0]
        processing_time = time.strftime("%Y%m%d-%H%M%S")
        sub_dir = os.path.join(debug_dir, f'{video_name}_{processing_time}')
        os.makedirs(sub_dir, exist_ok=True)
        # split video into scenes
        split_video_ffmpeg(video_path, scene_list, output_dir=sub_dir)
        logger.debug(f"{video_name}'s segmented scenes saved in {sub_dir}")

    return scene_list


def extract_frames(video_path, start_frame, end_frame):
    """
    Extracts frames from a video.
    Parameters
    ----------
    video_path : str
        Path to the video file.
    start_frame : int
        Start frame number.
    end_frame : int
        End frame number.
    Returns
    -------
    List of numpy arrays
        List of frames.
    """
    # clear memory
    clear_memory()

    # record processing time
    start_time = time.time()

    # read video
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            logger.error(f"Error reading frame {i} from {video_path}")
            break
    cap.release()

    # log processing time
    processing_time = time.time() - start_time
    logger.info(f"Frames({start_frame}, {end_frame}) extracted from {video_path} in {processing_time:.2f} seconds")

    return np.array(frames)




