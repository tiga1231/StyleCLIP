from base64 import b64encode
from io import BytesIO

import numpy as np
import torch
import torchvision
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from optimization.run_optimization import get_parser, main

# create Flask app
app = Flask(__name__)
CORS(app)

device = "cuda"


@app.route("/", methods=["GET"])
def index():
    return "Hello!"


def to_rgb(img):
    return (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()


def numpy2base64(x):
    """
    Convert numpy array to base64 string that is ready for html image src
    Parameters
    ----------
    x: numpy array
        x.shape == [h,w,3]
    Returns
    -------
    src: string
        a src string encoding image in base64
        ready for <img src='data:image/png;base64,'+src>
    """
    print("x.shape", x.shape)
    pil_img = Image.fromarray(x)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    string = b64encode(buff.getvalue()).decode("utf-8")
    return string


@app.route("/process_prompt", methods=["POST"])
def process_prompt():
    req = request.json
    prompt = req.get("prompt", "a woman in red hair")
    mode = req.get("mode", "edit")
    step = req.get("step", 30)

    # StyleCLIP process
    print("request:", req)

    print(f"Processing prompt '{prompt}'...")
    parser = get_parser()
    args = parser.parse_args()

    args.description = prompt
    args.id_lambda = 0
    args.l2_lambda = 0.006  # default=0.008
    args.mode = mode
    args.results_dir = "static"
    args.save_intermediate_image_every = 0
    args.step = step

    result_image = main(args)
    print(result_image.shape)

    # # Optionally, save image on disk
    # torchvision.utils.save_image(
    #     result_image.detach_().cpu(),
    #     f"{args.results_dir}/final_result.jpg",
    #     normalize=True,
    #     scale_each=True,
    #     range=(-1, 1),
    # )

    # prepare return
    result_image = result_image.detach_().cpu()  # [2,3,1024,1024]
    result_image = to_rgb(result_image)
    result_image_np = result_image.permute(0, 2, 3, 1).numpy()
    print(result_image_np.shape)
    return jsonify(
        dict(
            prompt=prompt,
            images=[numpy2base64(img_np) for img_np in result_image_np],
        )
    )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--embedding_fn',
    #     required=True,
    #     help='embedding file')
    # opt = parser.parse_args()

    app.run(host="0.0.0.0", port=9998, debug=True)
