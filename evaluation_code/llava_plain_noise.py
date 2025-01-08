import argparse
import os
import random



import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import json
import csv

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
    parser.add_argument("--model-path", type=str, default="ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")


    parser.add_argument("--image_file", type=str, default='/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/colorful_noise_image.png',
                        help="Image file")
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

image = load_image(args.image_file)
image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
print('[Initialization Finished]\n')





from llava_llama_2_utils import prompt_wrapper, generator


my_generator = generator.Generator(model=model, tokenizer=tokenizer)


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
        "prompt": "text_prompt"
    }))
    f.write("\n")

    for li in out:
        f.write(json.dumps(li))
        f.write("\n")