from base64 import b64encode
from io import BytesIO
from pathlib import Path
import os
import json

from tqdm import tqdm
# import numpy as np
import torch
import torchvision
from PIL import Image

from optimization.run_optimization import get_parser
from optimization.run_optimization import find_edit
from utils import ensure_checkpoint_exists
from models.stylegan2.model import Generator


def write(obj, filename):
    dir = Path(filename).parent
    if not dir.exists():
        os.makedirs(dir)
    with open(filename, 'w') as f:
        json.dump(obj, f)


if __name__ == '__main__':

    # set up constant, default args
    parser = get_parser()
    args = parser.parse_args()
    args.save_intermediate_image_every = 0
    args.id_lambda = 0.001
    args.l2_lambda = 0.008
    args.mode = 'edit'
    args.step = 100  # TODO increase me to ~100 in real experiment
    args.work_in_stylespace = False

    # args for this experiment
    latent_seeds = range(20)
    prompts = [
        'long nose',
        'short nose',
        'big eyes',
        'closed eyes',
        'old',
        'young',
        'red hair',
        'blue hair',
        'yellow hair',
        'short hair',
        'curly hair',
        'eyeglasses',
        'sunglasses',
    ]
    args.results_dir = "static/test_output"

    # load StyleGAN model
    ensure_checkpoint_exists(args.ckpt)
    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()

    write(prompts, f'{args.results_dir}/prompts.json')
    for i, seed in enumerate(latent_seeds):
        args.latent_seed = seed
        print(f'Latent seed {i}/{len(latent_seeds)}: "{seed}"')
        for j, prompt in enumerate(prompts):
            args.description = prompt
            print(f'Prompt {j}/{len(prompts)}: "{prompt}"')

            # StyleCLIP main process
            result = find_edit(g_ema, args)
            # for k, v in result.items():
            #     print(f'"{k}": shape={list(v.shape)}')

            # grab the StyleCLIP result
            result_image = result.get("final_result")  # [2,3,1024,1024]
            latent = result.get("latent")
            latent_code_init = result.get("latent_code_init")

            # Save latent code and the edited latent code, shape=[1,18,512]
            if j == 0:
                torch.save(latent_code_init,
                           f"{args.results_dir}/latent-code{i}.pth")
            torch.save(latent, f"{args.results_dir}/latent-code{i}-edit{j}.pth")

            # Optionally, save images on disk for inspection
            torchvision.utils.save_image(
                result_image.detach_().cpu(),
                f"{args.results_dir}/final_result-code{i}-edit{j}.jpg",
                normalize=True,
                scale_each=True,
                range=(-1, 1),
            )
