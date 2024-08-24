import gc
import torch

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.distributed.destroy_process_group()
    except:
        pass