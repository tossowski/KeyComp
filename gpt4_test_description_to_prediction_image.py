import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from datasets import load_dataset
from tqdm import tqdm

auth_token = os.environ["HF_TOKEN"]  # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))

print('Initialization Finished')
beam_size = 10
responses = []
for d in range(3):
    print("Looping through all the Winoground images...")
    for i, example in enumerate(tqdm(winoground)):
        #chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        example_id = example["id"]
        caption_0 = example["caption_0"]
        caption_1 = example["caption_1"]

        image_0 = example["image_0"].convert("RGB")
        image_1 = example["image_1"].convert("RGB")

    

        chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        image_list = []
        chat_responses = []
        conv = CONV_VISION.copy()

        with open(f"test/{example_id}_{0}/description.txt") as f:
            description_0  = f.read()[:200]

        with open(f"test/{example_id}_{1}/description.txt") as f:
            description_1  = f.read()[:200]
        
        #status = chat.upload_img(image_0, conv, image_list)
        chat.ask(f'Image A: {description_0}\nImage B: {description_1}\n\n Which image is more appropriate: "Image A" or "Image B"? Start your answer with A or B.', conv)
        out_text, out_token = chat.answer(conv=conv,
                                        img_list=image_list,
                                        num_beams=beam_size,
                                        temperature=1,
                                        max_new_tokens=50,
                                        max_length=2000)
        out_text = out_text.replace("\n", "\t")
        chat_responses.append(out_text)

        responses.append(f"{i},0|||No Response|||{chat_responses[0]}")


        chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        image_list = []
        chat_responses = []
        conv = CONV_VISION.copy()
        
        status = chat.upload_img(image_1, conv, image_list)

        with open(f"test/{example_id}_{1}/description.txt") as f:
            description = f.read()[:200]

        chat.ask(f'Image A: {description_0}\nImage B: {description_1}\n\n Which image is more appropriate: "Image A" or "Image B"? Start your answer with A or B.', conv)
        out_text, out_token = chat.answer(conv=conv,
                                        img_list=image_list,
                                        num_beams=beam_size,
                                        temperature=1,
                                        max_new_tokens=50,
                                        max_length=2000)

        out_text = out_text.replace("\n", "\t")
        chat_responses.append(out_text)
        responses.append(f"{i},1|||No Response|||{chat_responses[0]}")

    ##################################################################################

        

        with open(f"outputs/description_to_prediction_image_{d}.txt", "w") as f:
            for response in responses:
                f.write(f"{response}\n")