from base64 import b64encode
from io import BytesIO

# import numpy as np
import torch
# import torchvision
from PIL import Image

from optimization.run_optimization import get_parser
from optimization.run_optimization import find_edit
from utils import ensure_checkpoint_exists
from models.stylegan2.model import Generator

if __name__ == '__main__':

    # set args
    parser = get_parser()
    args = parser.parse_args()
    args.results_dir = "static"
    args.save_intermediate_image_every = 0
    args.description = "red hair"
    args.id_lambda = 0.0
    args.l2_lambda = 0.008
    args.latent_seed = 9
    args.mode = 'edit'
    args.step = 3
    args.work_in_stylespace = False

    ensure_checkpoint_exists(args.ckpt)
    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()

    # StyleCLIP main process
    result = find_edit(g_ema, args)
    for k, v in result.items():
        print(f'"{k}": shape={list(v.shape)}')

    # result_image = result.get("final_result")  # [2,3,1024,1024]
    # result_direction = result.get("latent_direction")  # [2,3,1024,1024]
