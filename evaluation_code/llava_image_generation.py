import argparse
import os


import random
import csv 
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import json
from tqdm import tqdm

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
    parser.add_argument("--model-path", type=str, default="ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")


    parser.add_argument("--image_dir", required=True, help="path to the directory containing visual inputs")
    parser.add_argument("--output_file", type=str, default='./result.jsonl',
                        help="Output file.")

    args = parser.parse_args()
    return args


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image



# ========================================
#             Model Initialization
# ========================================

print('>>> Initializing Models')

from llava_llama_2.utils import get_model
args = parse_args()

print('model = ', args.model_path)

tokenizer, model, image_processor, model_name = get_model(args)
model.eval()


print('[Initialization Finished]\n')


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image


from llava_llama_2_utils import prompt_wrapper, generator


my_generator = generator.Generator(model=model, tokenizer=tokenizer)


# ========================================
#             Inference
# ========================================

##  TODO: expose interface.
file = open("harmful_corpus/manual_harmful_instructions.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
prompts_eval = []
num = len(data)
for i in range(num):
    prompts_eval.append(data[i][0])



image_dir = args.image_dir
files = os.listdir(image_dir)
image_files = [file for file in files if file.endswith(".png") and not file.endswith(".bmp")]

# Sort the image files in alphabetical order

image_files = sorted(image_files)
out = []
with torch.no_grad():

    for j, image_file in enumerate(image_files):
        

        image_path = os.path.join(image_dir, image_file)
        image_path=load_image(image_path)
        image =image_processor.preprocess(image_path, return_tensors='pt')['pixel_values'].cuda()
        
        for i, user_message in enumerate(tqdm(prompts_eval)):

            print('################ Image %d, Question %d ################' % (j+1, i+1))
            print(" -- prompt: ---")
            print(user_message)
    
        

            text_prompt_template = prompt_wrapper.prepare_text_prompt(user_message)
            prompt = prompt_wrapper.Prompt(model, tokenizer, text_prompts=text_prompt_template, device=model.device)

            response = my_generator.generate(prompt, image)

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