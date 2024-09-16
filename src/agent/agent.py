import re
import os 
import json
import logging

from utils.prompts import QUERY_AGENTIC_PROMPT
from utils.clear_memory import clear_memory

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, mllm, video_processor):
        self.mllm = mllm    
        self.video_processor = video_processor

    @staticmethod
    def convert_time_to_seconds(time_str):
        pattern = r'(?:(\d+\.?\d*)h)?(?:(\d+\.?\d*)m)?(?:(\d+\.?\d*)s)?'
        match = re.match(pattern, time_str)
        hours = float(match.group(1)) if match.group(1) else 0
        minutes = float(match.group(2)) if match.group(2) else 0
        seconds = float(match.group(3)) if match.group(3) else 0
        return 3600 * hours + 60 * minutes + seconds
    
    def reformat_time_info(self, time_info):
        if not isinstance(time_info, list):
            return []
        if not isinstance(time_info[0], list):
            time_info = [time_info]
        result = []
        for t in time_info:
            try:
                if len(t) == 1:
                    result.append((self.convert_time_to_seconds(t[0]), ))
                else:
                    result.append((self.convert_time_to_seconds(t[0]), self.convert_time_to_seconds(t[1])))
            except Exception as e:
                logger.error(f"Failed to convert time info: {t}")
                continue
        return result

    def process_query(self, query):
        prompt = QUERY_AGENTIC_PROMPT.substitute(query=query)
        prompt = self.mllm.create_prompt(prompt)
        # get json response from the mllm
        json_str = self.mllm.generate(prompt)
        json_response = json_str.replace("\n", "").replace("```json", "").replace("```", "")
        # parse the json response
        time_info, need_semantic_search = [], True
        try:
            response = json.loads(json_response)
            time_info = response.get("time_info", [])
            need_semantic_search = response.get("need_semantic_search", True)
            time_info = self.reformat_time_info(time_info)
        except json.JSONDecodeError:
            logger.error("Failed to parse the JSON response")
        return time_info, need_semantic_search
    
    # time and semantic search
    def combined_search(self, video_id, query, **kwargs):
        time_info, need_semantic_search = self.process_query(query)
        visual_results, audio_results = [], []
        # time retrieval
        for t in time_info:
            if len(t) == 1:
                visual, audio = self.video_processor.search_video_by_time(video_id, t[0])
            else:
                visual, audio = self.video_processor.search_video_by_time(video_id, t[0], t[1])
            visual_results += visual
            audio_results += audio
        # semantic retrieval
        if need_semantic_search:
            v_top_k = kwargs.get("v_top_k", 5)
            a_top_k = kwargs.get("a_top_k", 5)
            visual, audio = self.video_processor.search_video_by_semantic(video_id, query, v_top_k, a_top_k)
            visual_results += visual
            audio_results += audio
        
        # remove duplicates
        unique_visual_results = []
        for v in visual_results:
            if v not in unique_visual_results:
                unique_visual_results.append(v)
        
        unique_audio_results = []
        for a in audio_results:
            if a not in unique_audio_results:
                unique_audio_results.append(a)

        return unique_visual_results, unique_audio_results
        
    


