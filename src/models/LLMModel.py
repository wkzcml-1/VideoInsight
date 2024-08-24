import os
import logging
import datetime
from abc import ABC, abstractmethod
import ray
import datetime

# vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.project_paths import CHECKPOINTS_DIR
from utils.project_paths import DEBUG_DIR
from utils.clear_memory import clear_memory
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

logger = logging.getLogger(__name__)

## LLM base class
class LLMModel(ABC):
    @abstractmethod
    def generate(self, prompts):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def unload_model(self):
        pass


## MiniCPM model
class MiniCPMModel(LLMModel):
    def __init__(self, **kwargs):
        path = kwargs.get('model_id', 'openbmb/MiniCPM-V_2_6_awq_int4')
        self.model_id = os.path.join(CHECKPOINTS_DIR, path)
        self.device = kwargs.get('device', 'cuda')
        self.debug = kwargs.get('debug', False)
        
        # vllm params
        self._gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 1)
        self._trust_remote_code = kwargs.get('trust_remote_code', True)
        self._max_model_len = kwargs.get('max_model_len', 2048)
        # vllm sampling params
        self._temperature = 0.5
        self._top_p = 0.8
        self._top_k = 100
        self._max_tokens = 1024

        # load model
        self.llm = None
        self.load_model()
        
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        
        self.stop_tokens = ['<|im_end|>', '<|endoftext|>']
        self.stop_token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in self.stop_tokens]

        # set sampling params
        self.set_sampling_params(**kwargs)

        logger.info(f"MiniCPM model loaded from {self.model_id}")

    def set_sampling_params(self, **kwargs):
        if 'temperature' in kwargs:
            self._temperature = kwargs.get('temperature')
        if 'top_p' in kwargs:
            self._top_p = kwargs.get('top_p')
        if 'top_k' in kwargs:
            self._top_k = kwargs.get('top_k')
        if 'max_tokens' in kwargs:
            self._max_tokens = kwargs.get('max_tokens')
        self.sampling_params = SamplingParams(
            stop_token_ids=self.stop_token_ids, 
            use_beam_search=False,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k, 
            max_tokens=self._max_tokens
        )
        logger.info(f"Sampling params set to {self.sampling_params}")
       
    def create_prompt(self, text, image=None, frames=None):
        # video prompt
        message = {"role": "user", "content": text}
        if frames and image:
            raise ValueError("Cannot provide both image and frames")
        elif frames:
            message["content"] = "".join(["(<image>./</image>)"] * len(frames)) + "\n" + text
        elif image:
            message["content"] = f"(<image>./</image>)\n{text}"
        prompt = self.tokenizer.apply_chat_template(
            [message],
            tokenize=False,
            add_generation_prompt=True
        )
        # generate prompt
        json_dict = {
            "prompt": prompt,
        }
        if frames:
            json_dict["multi_modal_data"] = { 
             "image": {
                "images": frames,
                "use_image_id": False,
                "max_slice_nums": 1 if len(frames) > 16 else 2
            }
        }
        elif image:
            json_dict["multi_modal_data"] = {
                "image": image
            }
        return json_dict

    def generate(self, prompts):
        if self.llm is None:
            self.load_model()
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        responses = [output.outputs[0].text for output in outputs]

        if self.debug:
            LLM_DEBUG_DIR = os.path.join(DEBUG_DIR, 'llm')
            os.makedirs(LLM_DEBUG_DIR, exist_ok=True)
            now_day = datetime.datetime.now().strftime("%Y-%m-%d")
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            debug_path = os.path.join(LLM_DEBUG_DIR, f'minicpm_{now_day}.txt')
            with open(debug_path, 'w') as f:
                # write timestamp first
                f.write(f"Timestamp: {now_time}\n")
                for response in responses:
                    f.write(f" {response}\n")

        return responses
    
    def generate_stream(self, prompts):
        pass

    def load_model(self):
        if self.llm is not None:
            return
        if 'cuda' in self.device:
            self.llm = LLM(model=self.model_id, device=self.device,     
                              trust_remote_code=self._trust_remote_code,
                              gpu_memory_utilization=self._gpu_memory_utilization,
                              max_model_len=self._max_model_len)
        else:
            self.llm = LLM(model=self.model_id, device=self.device, 
                           trust_remote_code=self._trust_remote_code, max_model_len=self._max_model_len)

    def unload_model(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.llm.llm_engine.model_executor
        del self.llm.llm_engine
        del self.llm
        clear_memory()
        ray.shutdown
        self.llm = None