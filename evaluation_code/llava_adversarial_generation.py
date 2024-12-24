import argparse
import os
from torchvision.utils import save_image
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-path", type=str, default="ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the GPU to load the model.")
    parser.add_argument("--n_iters", type=int, default=1000, help="specify the number of iterations for the attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step size of the attack")
    parser.add_argument("--constrained", default=False, action='store_true')
    parser.add_argument("--save-dir", type=str, default='output', help="save directory")
    parser.add_argument("--image-dir", type=str, default='images', help="directory containing input images")

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

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

import csv

file = open("harmful_corpus/derogatory_corpus.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
targets = []
num = len(data)
for i in range(num):
    targets.append(data[i][0])

print(targets)

from llava_llama_2_utils import visual_attacker

print('device = ', model.device)
my_attacker = visual_attacker.Attacker(args, model, tokenizer, targets, device=model.device, image_processor=image_processor)

image_files = os.listdir(args.image_dir)

for image_file in image_files:
    image_path = os.path.join(args.image_dir, image_file)
    image = load_image(image_path)
    image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()

    from llava_llama_2_utils import prompt_wrapper
    text_prompt_template = prompt_wrapper.prepare_text_prompt('')
    print(text_prompt_template)

    if not args.constrained:
        print('[unconstrained]')
        adv_img_prompt = my_attacker.attack_unconstrained(text_prompt_template, img=image, batch_size=8,
                                                         num_iter=args.n_iters, alpha=args.alpha / 255)
    else:
        adv_img_prompt = my_attacker.attack_constrained(text_prompt_template, img=image, batch_size=8,
                                                       num_iter=args.n_iters, alpha=args.alpha / 255,
                                                       epsilon=args.eps / 255)

    save_image(adv_img_prompt, os.path.join(args.save_dir, f"adversarial_{image_file}"))
    print(f"Adversarial image saved for {image_file}")

print('[Done]')