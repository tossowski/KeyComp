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
from matplotlib import pyplot as plt
import json
import os
from GQA.GQADataset import GQADataset
from torch.utils.data import DataLoader
from PIL import Image

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

print('Initializing Things')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

print('Initialization Finished')

def get_response(input_text, input_image):
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    conv = CONV_VISION.copy()
    image_list = []
    status = chat.upload_img(input_image, conv, image_list)
    chat.ask(input_text, conv)
    out_text, out_token = chat.answer(conv=conv,
                                    img_list=image_list,
                                    num_beams= 10,
                                    temperature=1,
                                    max_new_tokens=200,
                                    max_length=2000)
    return out_text

dataset = GQADataset("/data/ossowski/GQA", "test", f"cuda:{args.gpu_id}")
dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=2)


for i, batch in enumerate(tqdm(dataloader)):
    #print(batch['path_to_image'][0])
    if os.path.exists(batch['path_to_image'][0]):
        img = Image.open(batch['path_to_image'][0]).convert('RGB')
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        plt.title(batch["question"][0])
        
        out_text = get_response(batch['question'][0], img)
        plt.xlabel(f"Correct Answer: {batch['answer'][0]}\n" + f"Model Output: {out_text}")
        plt.savefig(f"GQA_output/test{i}.png")

        plt.close()
#description_0 = get_response("Describe the image in 1 sentence", image_0)
#description_1 = get_response("Describe the image in 1 sentence", image_1)
# for _ in range(10):


# details = []
# for i, shape in enumerate(image_1_annotations["shapes"]):
#     coords = shape["points"]
#     x1, y1 = list(map(int, coords[0]))
#     x2, y2 = list(map(int, coords[1]))
#     cropped = image_1.crop((x1, y1, x2, y2))
#     plt.imshow(cropped)
#     plt.savefig(f"{i}.png")
#     chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))

#     conv = CONV_VISION.copy()
#     image_list = []
#     status = chat.upload_img(cropped, conv, image_list)
#     out_text = get_response("Describe this object in detail with one sentence.", cropped)
#     out_text, out_token = chat.answer(conv=conv,
#                                     img_list=image_list,
#                                     num_beams= 10,
#                                     temperature=1,
#                                     max_new_tokens=50,
#                                     max_length=2000)
#     # center = f"({(x1 + x2) // 2},{(y1 + y2) // 2})"
#     # dimensions = f"({abs(x2 - x1)},{abs(y2 - y1)})"
#     details.append(f"At pixel {x1}, {y1} with width {abs(x2 - x1)} and height {abs(y2 - y1)}, {out_text}")

# for detail in details:
#     description_0 += " " + detail
    # print("-----------------------------------")
# conv = CONV_VISION.copy()
# status = chat.upload_img(image_0, conv, image_list)
# chat.ask("Describe the image in detail.", conv)
# out_text, out_token = chat.answer(conv=conv,
#                                 img_list=image_list,
#                                 num_beams=1,
#                                 temperature=0.01,
#                                 max_new_tokens=300,
#                                 max_length=2000)
# chat_responses.append(out_text.replace("\n", "\t"))
# chat.ask(f'Which of these 2 captions is more appropriate for the image: "{caption_0}" or "{caption_1}"? Answer in one sentence.', conv)
# out_text, out_token = chat.answer(conv=conv,
#                                 img_list=image_list,
#                                 num_beams=1,
#                                 temperature=0.01,
#                                 max_new_tokens=300,
#                                 max_length=2000)
# chat_responses.append(out_text.replace("\n", "\t"))
# responses.append(f"{i},0|||{chat_responses[0]}|||{chat_responses[1]}")


# chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))

# image_list = []
# chat_responses = []
# conv = CONV_VISION.copy()
# status = chat.upload_img(image_1, conv, image_list)
# chat.ask("Describe the image in detail.", conv)
# out_text, out_token = chat.answer(conv=conv,
#                                 img_list=image_list,
#                                 num_beams=1,
#                                 temperature=0.01,
#                                 max_new_tokens=300,
#                                 max_length=2000)
# chat_responses.append(out_text.replace("\n", "\t"))

# chat.ask(f'Which of these 2 captions is more appropriate for the image: "{caption_0}" or "{caption_1}"? Answer in one sentence.', conv)
# out_text, out_token = chat.answer(conv=conv,
#                                 img_list=image_list,
#                                 num_beams=1,
#                                 temperature=0.01,
#                                 max_new_tokens=300,
#                                 max_length=2000)


# chat_responses.append(out_text.replace("\n", "\t"))
# responses.append(f"{i},1|||{chat_responses[0]}|||{chat_responses[1]}")


# with open("outputs/responses.txt", "w") as f:
#     for response in responses:
#         f.write(f"{response}\n")