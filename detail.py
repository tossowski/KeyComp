import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
import spacy
import difflib
import random

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION
from PIL import Image
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
from textwrap import wrap
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import pickle
from scipy.ndimage.filters import gaussian_filter

auth_token = "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"  # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
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

def get_response(input_text, input_image, chat=None):
    if chat == None:
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
    return out_text, chat, conv, image_list

responses = []
print("Looping through all the Winoground images...")

#chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))

#nums = [72, 73, 74, 95, 96, 133, 149, 150, 164, 218, 221, 222, 224, 235, 237, 246, 274, 275, 321, 325, 326, 327, 332, 333, 334, 350, 364, 365, 398, 399]
#nums = [274, 275, 321, 325, 326, 327, 332, 333, 334, 350, 364, 365, 398, 399]
nums = [list(range(400))]
fname = "test2"
nlp = spacy.load("en_core_web_sm")
for num in nums:
    if num != 260:
        continue
    for d in range(5):
        for image_num in [0,1]:
            os.makedirs(f"{fname}/{num}_{image_num}", exist_ok=True)
            example = winoground[num]
            caption_0 = example['caption_0']
            caption_1 = example['caption_1']

            keywords = set()
            doc = nlp(caption_0)

            for token in doc:
                print(token.lemma_, token.pos_)
                if token.pos_ == "NOUN" or token.pos_ == "ADJ" or token.pos_ == "VERB":
                    keywords.add(token.lemma_)
            
            # with open(f"annotations/{2 * num}.json", "r") as f:
            #     image_0_annotations = json.load(f)
            # with open(f"annotations/{2 * num + 1}.json", "r") as f:
            #     image_1_annotations = json.load(f)


            # sam = sam_model_registry["default"](checkpoint="/data/ossowski/Segment_Anything/sam_vit_h_4b8939.pth")
            #mask_generator = SamAutomaticMaskGenerator(sam, points_per_side = 30, pred_iou_thresh=0.88, box_nms_thresh=0.9)

            #image = cv2.imread(f"../raw_images/{num * 2 + image_num}.png")
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #masks = mask_generator.generate(image)
            #sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
            #pickle.dump(sorted_anns, open(f"annotations/{num * 2 + image_num}.pkl", "wb"))

            # sentence1 = example["caption_0"].lower().split(" ")
            # sentence2 = example["caption_1"].lower().split(" ")
            # diff = []
            # diff2 = []
            # union = set()
            # for i, word in enumerate(sentence1):
            #     if sentence2[i] != word:
            #         if sentence2[i] not in union and word not in union:
            #             diff.append(word)
            #             diff2.append(sentence2[i])
            #             union.add(word)
            #             union.add(sentence2[i])

            # span1 = ' '.join(diff)
            # span2 = ' '.join(diff2)


            print(f"Describe this image using these keywords {keywords}")
        # random.shuffle()
            # sentence = caption_0.split()
            # random.shuffle(sentence)
            description_0, chat, conv, image_list = get_response(f"Describe this image using these keywords {keywords}", example[f'image_{image_num}'].convert('RGB'))
            print(description_0)

            #out_text, _, _, _ = get_response("Select the best caption for this image:\nA:\"{caption_0}\"\nB:\"{caption_1}\"\nThink step-by-step and explain your answer in 1-2 sentences. Start your answer with A or B", None, chat)
            # chat.ask("Select the best caption for this image:\nA:\"{caption_0}\"\nB:\"{caption_1}\"\nThink step-by-step and explain your answer in 1-2 sentences. Start your answer with A or B", conv)
            # out_text, out_token = chat.answer(conv=conv,
            #                                 img_list=image_list,
            #                                 num_beams= 10,
            #                                 temperature=1,
            #                                 max_new_tokens=50,
            #                                 max_length=2000)
            # print(out_text)
            # doc = nlp(description_0)

            # for token in doc:
            #     if token.pos_ == "NOUN":
            #         keywords.add(token.lemma_)
            
            # with open(f"{fname}/{num}_{image_num}/description_{d}.txt", "w") as f:
            #     f.write(description_0)

        # visited = np.zeros_like(sorted_anns[0]['segmentation'])
        # h, w= visited.shape
        # masks = []
        # cutoff = 500
        # for i, mask in enumerate(sorted_anns):
        #     #print(mask['area'], h * w * 0.5)
        #     #if mask['area'] < 10000 or mask['area'] > h * w * 0.5:
        #     if mask['area'] < 10000 or mask['area'] > h * w * 0.5:

        #         continue
        #     bool_mask = mask['segmentation']
        #     #total_overlap = 0
        #     total_overlap = np.sum(visited * bool_mask, axis=None)
        #     print(total_overlap)
        #     if total_overlap > cutoff:

        #         continue
       
        #     visited += bool_mask
        #     visited[visited > 1] = 1
        #     masks.append(i)


        # details = []
        # for k in masks:
        #     bool_mask = sorted_anns[k]['segmentation']
        #     x1, x2, y1, y2 = len(bool_mask), 0, len(bool_mask[0]), 0
        #     for i in range(len(bool_mask)):
        #         for j in range(len(bool_mask[0])):
        #             if bool_mask[i][j] == 1:
        #                 x1 = min(x1, i)
        #                 x2 = max(x2, i)
        #                 y1 = min(y1, j)
        #                 y2 = max(y2, j)
        #     test = Image.open(f"../raw_images/{num * 2 + image_num}.png").convert('RGB')
            
        #     im_copy  = np.array(test)
        #     im_copy[~bool_mask] = np.array([255,255,255])
        #     cropped = Image.fromarray(im_copy).crop((y1, x1, y2, x2))
        #     m = np.zeros((max(np.array(cropped).shape), max(np.array(cropped).shape), 3), dtype=np.uint8)
        #     if np.array(cropped).shape[0] <= np.array(cropped).shape[1]:
        #         offset = abs(x2 - x1) // 2
        #         start = m.shape[0] // 2 - offset 
        #         m[start:start + x2 - x1, :, :] = np.array(cropped)
        #     else:
        #         offset = (y2 - y1) // 2
        #         start = m.shape[1] // 2 - offset
        #         m[:, start:start+y2 - y1, :] = np.array(cropped)

            

        #     chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))

        #     conv = CONV_VISION.copy()
        #     image_list = []
        #     status = chat.upload_img(Image.fromarray(m), conv, image_list)
        #     out_text = get_response("Describe the image.", cropped)
        #     out_text, out_token = chat.answer(conv=conv,
        #                                     img_list=image_list,
        #                                     num_beams= 10,
        #                                     temperature=1,
        #                                     max_new_tokens=100,
        #                                     max_length=2000)
        #     out_text = out_text.replace("\n", "\t")
        #     print(out_text)
        #     doc = nlp(out_text)
        #     #print(keywords)
        #     relevant = False
        #     for token in doc:
        #         if token.pos_ == "NOUN":
        #             #print(token.lemma_)
        #             if token.lemma_ == "image":
        #                 continue
        #             if token.lemma_ in keywords:
        #                 relevant = True
            
        #     if relevant:
        #         plt.imshow(m)
        #         plt.xlabel("\n".join(wrap(out_text, 60)))
        #         plt.savefig(f"{fname}/{num}_{image_num}/{k}.png")
        #         details.append(f"{x1},{y1},{abs(x2 - x1)},{abs(y2 - y1)}|||{out_text}")
        # with open(f"{fname}/{num}_{image_num}/details.txt", "w") as f:
        #     for detail in details:
        #         f.write(detail + "\n")
# for detail in details:
#     description_0 += " " + detail
# print(description_0)
# print(get_response(description_0  + "\n\n" + f'Which caption is more appropriate: "{caption_0}" or "{caption_1}"? Answer with 1 sentence.', example[f'image_{image_num}']))

    #plt.imshow(m)

    

# example_id = example["id"]
# caption_0 = example["caption_0"]
# caption_1 = example["caption_1"]

# image_0 = example["image_0"].convert("RGB")
# image_1 = example["image_1"].convert("RGB")

# description_0 = get_response("Describe the image in 1 sentence", image_0)


# details = []
# for i, shape in enumerate(image_1_annotations["shapes"]):
#     coords = shape["points"]
#     x1, y1 = list(map(int, coords[0]))
#     x2, y2 = list(map(int, coords[1]))
#     cropped = image_1.crop((x1, y1, x2, y2))
#     im_copy  = np.array(image_0)


#     mask = np.zeros((max(np.array(cropped).shape), max(np.array(cropped).shape), 3), dtype=np.uint8)
#     if np.array(cropped).shape[0] <= np.array(cropped).shape[1]:
#         offset = abs(y2 - y1) // 2
#         start = mask.shape[0] // 2 - offset 
#         mask[start:start + y2 - y1, :, :] = np.array(cropped)
#     else:
#         offset = (x2 - x1) // 2
#         start = mask.shape[1] // 2 - offset
#         mask[:, start:start+x2 - x1, :] = np.array(cropped)
    
    
#     chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))

#     conv = CONV_VISION.copy()
#     image_list = []
#     status = chat.upload_img(Image.fromarray(mask), conv, image_list)
#     out_text = get_response("One object is in this picture. Describe it in detail with one sentence.", cropped)
#     out_text, out_token = chat.answer(conv=conv,
#                                     img_list=image_list,
#                                     num_beams= 10,
#                                     temperature=1,
#                                     max_new_tokens=50,
#                                     max_length=2000)
    
#     plt.imshow(mask)
#     plt.xlabel("\n".join(wrap(out_text, 60)))
#     plt.savefig(f"{i}.png")
#     details.append(f"At pixel {x1}, {y1} with width {abs(x2 - x1)} and height {abs(y2 - y1)}, {out_text}")

# for detail in details:
#     description_0 += " " + detail
# print(description_0 + "\n\n" + f"Which caption is more appropriate: {caption_0} or {caption_1}? Answer with 1 sentence.")
# print(get_response(description_0  + "\n\n" + f'Which caption is more appropriate: "{caption_0}" or "{caption_1}"? Answer with 1 sentence.', image_0))
 