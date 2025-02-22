import argparse
import os

import random
import csv
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

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


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--image_dir", required=True, help="path to the directory containing visual inputs")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the GPU to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--output_file", type=str, default='./result.jsonl',
                        help="Output file.")
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

# ========================================
#             Gradio Setting
# ========================================

# init chat state
def upload_img(img):
    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_img(img, chat_state, img_list)
    return chat_state, img_list


def ask(user_message, chat_state):
    chat.ask(user_message, chat_state)
    return chat_state


def answer(chat_state, img_list, num_beams=1, temperature=1.0):
    llm_message  = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]

    return llm_message, chat_state, img_list


file = open("harmful_corpus/manual_harmful_instructions.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
prompts_eval = []
num = len(data)
for i in range(num):
    prompts_eval.append(data[i][0])

image_dir = args.image_dir

files = os.listdir(image_dir)
image_files = [file for file in files if file.endswith("png.bmp")]
image_files = sorted(image_files)

out = []

print("######## Start Chatting ########")

with torch.no_grad():

    for j, image_file in enumerate(image_files):

        image_path = os.path.join(image_dir, image_file)
        img = Image.open(image_path).convert('RGB')
        img = vis_processor(img).unsqueeze(0).to(model.device)

        for i, prompt_to_eval in enumerate(tqdm(prompts_eval)):

            user_message = prompt_to_eval
            chat_state, img_list = upload_img(img)

            print('################ Image %d, Question %d ################' % (j+1, i+1))
            chat_state = ask(user_message, chat_state)
            llm_message, chat_state, img_list = answer(chat_state,img_list)

            print('>>> User:', user_message)
            print('\n')

            print('>>> LLM:\n')
            print(llm_message)
            print('\n\n')

            out.append({'prompt': user_message, 'continuation': llm_message})
            print()

            # Record the conversation history
           




with open(args.output_file, 'w') as f:
    f.write(json.dumps({
        "args": vars(args),
        "prompt": "text_prompt"
    }))
    f.write("\n")

    for li in out:
        f.write(json.dumps(li))
        f.write("\n")

print("Conversation history saved to", args.output_file)