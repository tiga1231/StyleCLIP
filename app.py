from base64 import b64encode
from io import BytesIO

# import numpy as np
import torch
# import torchvision
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from optimization.run_optimization import get_parser
from optimization.run_optimization import main as find_directions
from optimization.run_optimization import load_generator

# create Flask app
app = Flask(__name__)
CORS(app)

device = "cuda"


@app.route("/", methods=["GET"])
def index():
    return "Hello!"


def to_rgb(img):
    return (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()


def encode_numpy(array):
    return b64encode(array.tobytes()).decode()


def encode_numpy_img(array):
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
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    string = b64encode(buff.getvalue()).decode("utf-8")
    return string


# globals
g_ema = load_generator()


@app.route("/get_weighted_images", methods=["GET", "POST"])
def get_weighted_images():
    global g_ema
    if request.method == 'GET':  # testing only
        latent = torch.load('static/test_output/latent-code0.pth')

    else:
        req = request.get_json()
        code_index = req["code_index"]
        direction_indices = req["direction_indices"]
        weights = req["weights"]
        weights = torch.tensor(
            weights,
            dtype=torch.float32,
        )
        print(weights)

        # (lazy) load latent edits from file
        w0 = torch.load(f'static/test_output/latent-code{code_index}.pth',
                        map_location='cpu')
        # latent = w0
        w1 = [
            torch.load(
                f'static/test_output/latent-code{code_index}-edit{di}.pth',
                map_location='cpu') for di in direction_indices
        ]
        w1 = torch.cat(w1, 0)
        latent = (weights @ w1.view(-1, 18 * 512)).view(-1, 18, 512)
        print(latent.shape)
        # combine latent codes based on weights
        # generate image(s)
        # image_strings = []  # TODO

    latent = latent.to(device)
    img_gen, _ = g_ema(
        [latent],
        input_is_latent=True,
        randomize_noise=False,
        input_is_stylespace=False,
    )
    print(img_gen.shape)
    # torch tensor to numpy
    img_gen = img_gen.detach_().cpu()  # [2,3,1024,1024]
    img_gen = to_rgb(img_gen)
    img_gen = img_gen.permute(0, 2, 3, 1).numpy()
    image_strings = [encode_numpy_img(img_gen[0])]

    return jsonify(dict(images=image_strings))


@app.route("/process_prompt", methods=["POST"])
def process_prompt():
    # Get user request data
    req = request.json
    print("request:", req)
    prompt = req.get("prompt", "a woman in red hair")
    mode = req.get("mode", "edit")
    step = req.get("step", 30)
    id_lambda = req.get("id_lambda", 0.0)
    l2_lambda = req.get("l2_lambda", 0.008)
    latent_seed = req.get("latent_seed", 0)
    work_in_stylespace = req.get("work_in_stylespace", False)
    print(f"Processing prompt '{prompt}'...")

    # set args
    parser = get_parser()
    args = parser.parse_args()
    args.results_dir = "static"
    args.save_intermediate_image_every = 0
    args.description = prompt
    args.id_lambda = id_lambda
    args.l2_lambda = l2_lambda
    args.latent_seed = latent_seed
    args.mode = mode
    args.step = step
    args.work_in_stylespace = work_in_stylespace

    # StyleCLIP main process
    result = find_directions(args)
    result_image = result.get("final_result")  # [2,3,1024,1024]

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
    return jsonify(
        dict(
            prompt=prompt,
            images=[encode_numpy_img(img_np) for img_np in result_image_np],
        ))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--embedding_fn',
    #     required=True,
    #     help='embedding file')
    # opt = parser.parse_args()

    app.run(host="0.0.0.0", port=9998, debug=True)
