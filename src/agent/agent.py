import re
import os 
import json
import logging

from utils.prompts import QUERY_AGENTIC_PROMPT
from utils.clear_memory import clear_memory

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, mllm):
        self.mllm = mllm

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
        result = []
        for t in time_info:
            if len(t) == 1:
                result.append(self.convert_time_to_seconds(t[0]))
            else:
                result.append((self.convert_time_to_seconds(t[0]), self.convert_time_to_seconds(t[1])))
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
        
    


