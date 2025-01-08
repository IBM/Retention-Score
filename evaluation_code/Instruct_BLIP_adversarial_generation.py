import argparse
import os


import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image

from lavis.models import load_model_and_preprocess
from blip_utils import visual_attacker


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the GPU to load the model.")
    parser.add_argument("--n_iters", type=int, default=5000, help="specify the number of iterations for the attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step size of the attack")
    parser.add_argument("--constrained", default=False, action='store_true')
    parser.add_argument("--save-dir", type=str, default='output', help="save directory")
    parser.add_argument("--image-dir", type=str, default='images', help="directory containing input images")

    args = parser.parse_args()
    return args


print('>>> Initializing Models')

args = parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model, vis_processor, _ = load_model_and_preprocess(
    name='blip2_vicuna_instruct',
    model_type='vicuna13b',
    is_eval=True,
    device=device,
)
model.eval()

print('[Initialization Finished]\n')


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

import csv

file = open("harmful_corpus/derogatory_corpus.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
targets = []
num = len(data)
for i in range(num):
    targets.append(data[i][0])


my_attacker = visual_attacker.Attacker(args, model, targets, device=model.device, is_rtp=False)

image_files = os.listdir(args.image_dir)

for image_file in image_files:
    image_path = os.path.join(args.image_dir, image_file)
    img = Image.open(image_path).convert('RGB')
    img = vis_processor["eval"](img).unsqueeze(0).to(device)

    if not args.constrained:
        adv_img_prompt = my_attacker.attack_unconstrained(img=img, batch_size=8,
                                                          num_iter=args.n_iters, alpha=args.alpha / 255)
    else:
        adv_img_prompt = my_attacker.attack_constrained(img=img, batch_size=8,
                                                        num_iter=args.n_iters, alpha=args.alpha / 255,
                                                        epsilon=args.eps / 255)

    save_image(adv_img_prompt, os.path.join(args.save_dir, f"adversarial_{image_file}"))
    print(f"Adversarial image saved for {image_file}")

print('[Done]')