# Use a pipeline as a high-level helper
from transformers import pipeline



from transformers import AutoTokenizer, AutoModelForCausalLM
from minigpt_utils import prompt_wrapper, generator

prefix = ""
prefix += "###Human:%s ###Assistant:"

tokenizer = AutoTokenizer.from_pretrained("/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/weight")
model = AutoModelForCausalLM.from_pretrained("/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/weight",load_in_8bit=True,device_map="auto")

#tokenizer = AutoTokenizer.from_pretrained("Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/ckpts/vicuna-13b-v1.1")
#model = AutoModelForCausalLM.from_pretrained("Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/ckpts/vicuna-13b-v1.1",load_in_8bit=True,device_map="auto")


def rtp_read(text_file):
    with open(text_file, 'r') as f:
        datasets = []
        for row in f:

           
            recover = json.loads(row)['recover'].strip()
            a=len(recover)
            datasets.append(recover[6:(a-5)])
        
  
       
        
    return datasets

import json


datasets= rtp_read('paraphrasing.json')

datasets = datasets[:1000] # shortlist 1k

print('size of shortlisted dataset : ', len(datasets))

from tqdm import tqdm



import torch
# Generate


# Generate




with torch.no_grad():
    for i, data in tqdm(enumerate(datasets)):

        user_message = data

        print(f" ----- {i} ----")
        print(" -- prompt: ---")
        #user_message="test something?"
        print(user_message)
        print(data)
       
       
        inputs = tokenizer([prefix % user_message], return_tensors="pt")
        generate_ids = model.generate(inputs.input_ids.to(0), max_new_tokens=300, use_nucleus_sampling=True,temperature=1.0,top_p=0.9)
        print(generate_ids)
        a=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print(" -- response: ---")
        print(a)




