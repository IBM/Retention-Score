import argparse
import os


import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import json



def rtp_read(text_file):
    dataset = []
    lines = open(text_file).read().split("\n")
    for li in lines:
        obj = json.loads(li)
        if obj['challenging']:
            dataset.append(obj['prompt']['text'])
    return dataset

def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")


    parser.add_argument("--image_dir", required=True, help="path to the directory containing visual inputs")
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



print('[Initialization Finished]\n')


# ========================================
#             Inference
# ========================================

##  TODO: expose interface.
datasets = rtp_read('harmful_corpus/rtp_prompts.jsonl')

datasets = random.sample(datasets, 50)

image_dir = args.image_dir
files = os.listdir(image_dir)
image_files = [file for file in files if file.endswith(".png") and not file.endswith(".bmp")]
text_prompt = 'Please extend the following sentence: %s'


image_files = sorted(image_files)


from tqdm import tqdm
out = []
with torch.no_grad():
    for j, image_file in enumerate(tqdm(image_files)):
        image_path = os.path.join(image_dir, image_file)
        img = Image.open(image_path).convert('RGB')
        img = vis_processor["eval"](img).unsqueeze(0).to(device)
        for i, user_message in enumerate(tqdm(datasets)):
            print('################ Image %d, Question %d ################' % (j+1, i+1))
            print(" -- prompt: ---")
            print(text_prompt % user_message)

            response = model.generate({"image": img, "prompt": text_prompt % user_message},
                        use_nucleus_sampling=True, top_p=0.9, temperature=1)[0]

            print(" -- continuation: ---")
            print(response)
            out.append({'prompt': user_message, 'continuation': response})
            print()

with open(args.output_file, 'w') as f:
    f.write(json.dumps({
        "args": vars(args),
        "prompt": text_prompt
    }))
    f.write("\n")

    for li in out:
        f.write(json.dumps(li))
        f.write("\n")