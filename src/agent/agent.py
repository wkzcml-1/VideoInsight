import re
import os 
import json
import logging

from utils.prompts import QUERY_AGENTIC_PROMPT
from utils.prompts import QUERY_REWRITE_PROMPT
from utils.prompts import VIDEO_INSIGHT_PROMPT
from utils.prompts import VIDEO_SUMMARY_PROMPT
from utils.clear_memory import clear_memory

from utils.load_config import load_config

config = load_config()

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, mllm, video_processor, **kwargs):
        self.mllm = mllm    
        self.video_processor = video_processor
        self.chat_history = []
        # max chat history turn
        self.max_turn = kwargs.get("MAX_TURN", 10)

    @staticmethod
    def convert_time_to_seconds(time_str):
        pattern = r'(?:(\d+\.?\d*)h)?(?:(\d+\.?\d*)m)?(?:(\d+\.?\d*)s)?'
        match = re.match(pattern, time_str)
        if match:
            hours = float(match.group(1)) if match.group(1) else 0
            minutes = float(match.group(2)) if match.group(2) else 0
            seconds = float(match.group(3)) if match.group(3) else 0
        else:
            hours, minutes, seconds = 0, 0, float(time_str)
        return 3600 * hours + 60 * minutes + seconds
    
    def reformat_time_info(self, time_info):
        if not isinstance(time_info, list):
            return []
        if not time_info:
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
    
    def rewrite_query(self, query):
        clear_memory()
        history = self.chat_history
        if len(self.chat_history) > self.max_turn:
            history = self.chat_history[-self.max_turn:]
        prompt = QUERY_REWRITE_PROMPT.substitute(query=query, history=history)
        prompt = self.mllm.create_prompt(prompt)
        result = self.mllm.generate(prompt)
        logger.info(f"{query} rewritten query: {result}")
        return result

    def process_query(self, query):
        clear_memory()
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
        logger.info(f"Time info: {time_info}, need semantic search: {need_semantic_search}")
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
            visual, audio = self.video_processor.search_video_by_semantic(video_id, query, **kwargs)
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

        # sort the results
        sorted_visual_results = sorted(unique_visual_results, key=lambda x: x['segment_id'])
        sorted_audio_results = sorted(unique_audio_results, key=lambda x: x['segment_id'])

        return sorted_visual_results, sorted_audio_results
    
    def generate_video_basic_info(self, video_id):
        video_metadata = self.video_processor.search_video_metadata(video_id)
        if not video_metadata:
            return None
        return {
            'title': video_metadata['name'],
            'duration': self.reformat_time(video_metadata['duration']),
        }
    
    @staticmethod
    def reformat_time(time):
        # time: seconds 
        hour = int(time // 3600)
        minute = int((time - hour * 3600) // 60)
        second = round(time - hour * 3600 - minute * 60, 1)
        return f"{hour}:{minute}:{second}"
    
    def generate_timeline(self, recalls, text_field='description'):
        timeline = ""
        for r in recalls:
            start_time = self.reformat_time(r['start_time'])
            end_time = self.reformat_time(r['end_time'])
            timeline += f"{start_time} - {end_time}: \n{r[text_field]}\n"
        return timeline
    
    def video_chat(self, video_id, query):
        # video basic info
        video_basic_info = self.generate_video_basic_info(video_id)
        if not video_basic_info:
            return "视频不存在"
        # rewrite query
        rewritten_query = self.rewrite_query(query)
        # search video
        kwargs = {
            'v_top_k': config['VIDEO_PROCESSING']['VISUAL_RECALL_NUM'],
            'a_top_k': config['VIDEO_PROCESSING']['AUDIO_RECALL_NUM'],
        }
        visual_results, audio_results = self.combined_search(video_id, rewritten_query, **kwargs)
        # generate timeline
        visual_timeline = self.generate_timeline(visual_results, text_field='description')
        audio_timeline = self.generate_timeline(audio_results, text_field='transcript')
        # generate response
        prompt = VIDEO_INSIGHT_PROMPT.substitute(
            video_basic_info=video_basic_info,
            scene_timeline=visual_timeline,
            audio_timeline=audio_timeline,
            query=query
        )
        # generate response
        mllm_prompt = self.mllm.create_prompt(prompt)
        response = self.mllm.generate(mllm_prompt)
        clear_memory()
        self.chat_history.append({'user': query, 'assistant': response})
        return response
    
    def progressive_summary(self, video_id):
        # judge whether the video summary has been generated
        video_summary = self.video_processor.get_video_summary(video_id)
        if video_summary:
            return video_summary['summary']

        # make sure how many scenes in the video
        scenes_count = self.video_processor.get_num_of_visual_segments(video_id)
        if not scenes_count:
            logger.error(f"Video({video_id}) has no scenes")
            return None
        # iterate over each scene
        past_summary = ""
        past_start_time, past_end_time = 0, 0
        for scene_id in range(1, scenes_count + 1):
            visual_info = self.video_processor.get_visual_segments(video_id, scene_id)
            if not visual_info:
                logger.error(f"Scene({scene_id}) not found in video({video_id})")
                continue
            # get min start time and max end time in visual info
            start_time = visual_info.get('start_time')
            end_time = visual_info.get('end_time')
            audio_info = self.video_processor.search_video_by_time(video_id, start_time, end_time, collection='audio')
            scene_timeline = self.generate_timeline([visual_info], text_field='description')
            audio_timeline = self.generate_timeline(audio_info, text_field='transcript')
            # extract key frame
            key_frame = self.video_processor.extract_key_frame(video_id, start_time, end_time)
            # generate prompt
            prompt = VIDEO_SUMMARY_PROMPT.substitute(
                past_summary="" if not past_summary else f"{past_start_time} - {past_end_time}: \n{past_summary}\n",
                scene_timeline=scene_timeline,
                audio_timeline=audio_timeline
            )
            # generate response
            mllm_prompt = self.mllm.create_prompt(prompt, image=key_frame)
            past_summary = self.mllm.generate(mllm_prompt)
            past_end_time = end_time
            # write summary to database
            summary_dict = {'summary': past_summary, 'start_time': past_start_time, 'end_time': past_end_time}
            self.video_processor.insert_summary(video_id, scene_id, summary_dict)

        return past_summary

    def clear_history(self):
        del self.chat_history
        clear_memory()
        self.chat_history = []
        logger.info("Chat history cleared")

    def delete_video(self, video_id):
        self.video_processor.delete_by_video_id(video_id)

    def drop_all_video_info(self):
        self.video_processor.drop_all()
        



        
        
    


