import argparse
import os

import random
import csv

from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image
from minigpt_utils import visual_attacker, prompt_wrapper

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the GPU to load the model.")
    parser.add_argument("--n_iters", type=int, default=500, help="specify the number of iterations for the attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget.")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack.")
    parser.add_argument("--constrained", default=False, action='store_true')
    parser.add_argument("--save_dir", type=str, default='output', help="save directory.")
    parser.add_argument("--image_dir", type=str, default='images', help="directory containing input images.")

    parser.add_argument("--options", nargs="+",
                        help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecate), change to --cfg-options instead.")
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# Model Initialization

print('>>> Initializing Models')

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

print('[Initialization Finished]')


if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)


def process_images(image_dir):
    image_files = os.listdir(image_dir)
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        if os.path.isfile(image_path):
            img = Image.open(image_path).convert('RGB')
            img = vis_processor(img).unsqueeze(0).to(model.device)

            text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt_no_text_input

            if not args.constrained:
                adv_img_prompt = my_attacker.attack_unconstrained(text_prompt_template,
                                                                  img=img, batch_size=8,
                                                                  num_iter=5000, alpha=args.alpha / 255)
            else:
                adv_img_prompt = my_attacker.attack_constrained(text_prompt_template,
                                                                img=img, batch_size=8,
                                                                num_iter=5000, alpha=args.alpha / 255,
                                                                epsilon=args.eps / 255)

            save_image(adv_img_prompt, os.path.join(args.save_dir, f'bad_prompt_{image_file}.bmp'))
            print(f'Processed image: {image_file}')


import csv

file = open("harmful_corpus/derogatory_corpus.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
targets = []
num = len(data)
for i in range(num):
    targets.append(data[i][0])

my_attacker = visual_attacker.Attacker(args, model, targets, device=model.device, is_rtp=False)

image_dir = args.image_dir
process_images(image_dir)

print('[Done]')