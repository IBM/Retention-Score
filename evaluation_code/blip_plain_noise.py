import argparse
import os


import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import json
import csv
from tqdm import tqdm

'''
def rtp_read(text_file):
    with open(text_file, 'r') as f:
        datasets = []
       
        reader = csv.DictReader(f)
        for row in reader:
            goal = row['goal']
            target = row['target']
            datasets.append(goal)
        
        
    return datasets
'''

def rtp_read(text_file):
    with open(text_file, 'r') as f:
        datasets = []
        for row in f:

           
            recover = json.loads(row)['recover'].strip()
            a=len(recover)
            datasets.append(recover)
        
  
       
        
    return datasets

def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")


    parser.add_argument("--image_file", type=str, default='/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/colorful_noise_image.png',
                        help="Image file")
    parser.add_argument("--output_file", type=str, default='./result.jsonl',
                        help="Output file.")

    args = parser.parse_args()
    return args



# ========================================
#             Model Initialization
# ========================================

print('>>> Initializing Models')

from lavis.models import load_model_and_preprocess

args = parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# remember to modify the parameter llm_model in ./lavis/configs/models/blip2/blip2_instruct_vicuna13b.yaml to the path that store the vicuna weights
model, vis_processor, _ = load_model_and_preprocess(
        name='blip2_vicuna_instruct',
        model_type='vicuna13b',
        is_eval=True,
        device=device,
    )
model.eval()
"""
Source code of the model in:
    ./lavis/models/blip2_models/blip2_vicuna_instruct.py
"""

img = Image.open(args.image_file).convert('RGB')
img = vis_processor["eval"](img).unsqueeze(0).to(device)

print('[Initialization Finished]\n')



prefix = 'Please answer the following questions: %s'


#prefix  = "<s>[INST] %s[/INST]</s>"

# ========================================
#             Inference
# ========================================

##  TODO: expose interface.
#datasets = rtp_read('harmful_corpus/rtp_prompts.jsonl')
datasets= rtp_read('/AutoDAN/20_50_modified_merged_file.jsonl')

datasets = datasets # shortlist 1k

print('size of shortlisted dataset : ', len(datasets))

out = []
with torch.no_grad():
    for i, user_message in enumerate(datasets):
        print(f" ----- {i} ----")
        print(" -- prompt: ---")
        print(user_message)

        #response = model.generate({"image": img, "prompt":prefix % user_message},
                       #use_nucleus_sampling=True, top_p=0.9, temperature=1)[0]
        response = model.generate({"image": img, "prompt":prefix % user_message},
                       use_nucleus_sampling=True, top_p=0.9, temperature=1,repetition_penalty=10.0)[0]
        print(" -- continuation: ---")
        print(response)
        out.append({'prompt': user_message, 'continuation': response})
        print()

with open(args.output_file, 'w') as f:
    f.write(json.dumps({
        "args": vars(args),
        "prompt": "text_prompt"
    }))
    f.write("\n")

    for li in out:
        f.write(json.dumps(li))
        f.write("\n")