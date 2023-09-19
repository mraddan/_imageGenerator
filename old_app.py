import flask
import transformers
import accelerate
from flask import Flask, flash, redirect, render_template, request, Request, jsonify

import re
import torch
import json
import os
import base64

from io import BytesIO

from diffusers import DiffusionPipeline, KDPM2DiscreteScheduler

app = Flask(__name__, template_folder="templates")

model = "stabilityai/stable-diffusion-xl-base-1.0"
model2 = "stabilityai/stable-diffusion-xl-refiner-1.0"
sch = KDPM2DiscreteScheduler.from_pretrained(model, subfolder="scheduler")
sch1 = KDPM2DiscreteScheduler.from_pretrained(model2, subfolder="scheduler")
base = DiffusionPipeline.from_pretrained(
    model, torch_dtype=torch.float16, variant="fp16", scheduler=sch
)

refiner = DiffusionPipeline.from_pretrained(
    model2,
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    scheduler=sch1,
    use_safetensors=True,
    variant="fp16",
)

base.to("cuda")
refiner.to("cuda")

base.enable_model_cpu_offload()
refiner.enable_model_cpu_offload()

base.enable_xformers_memory_efficient_attention()
refiner.enable_xformers_memory_efficient_attention()

n_steps = 40

image_counter = 0

np = "double face, hands, wrist, Ugly, Duplicate, Extra fingers, Mutated hands, Poorly drawn face, Mutation, Deformed, Blurry, Bad anatomy, Bad proportions, Extra limbs, cloned face, Disfigured, Missing arms, Missing legs, Extra arms, Extra legs, Fused fingers, Too many fingers, Long neck, writing, letters, Multiple bodies, multiple heads, extra hands, extra fingers, ugly, skinny, extra leg, extra foot, blur, bad anatomy, double body, stacked body, fused hands, fused body, fused heads, fused legs, fused feet, multiple faces, ((conjoined)), (siamese twin), double faces, two faces, texts, watermarked, watermark, logo, face out of frame, stacked background, ((out of frame portrait)), bucktoothed, cropped"


def sanitize_input(input_str):
    return re.sub(r"\W+", "", input_str)


@app.route("/", methods=["GET", "POST"])
def generate_image():
    global image_counter

    if request.method == "POST":
        prompt = request.form.get("prompt")
        style = request.form.get("style")
        style2 = request.form.get("style2")
        height = request.form.get("height")
        width = request.form.get("width")

        if height is None or width is None:
            return jsonify({"error": "Height and width must be provided."}), 400

        try:
            height = int(height)
            width = int(width)
        except ValueError:
            return jsonify({"error": "Height and width must be valid integers."}), 400

        image = base(
            prompt=prompt + style + style2,
            negative_prompt=np,
            num_inference_steps=n_steps,
            height=height,
            width=width,
            guidance_scale=7,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt + style + style2,
            negative_prompt=np,
            num_inference_steps=n_steps,
            guidance_scale=7,
            image=image,
        ).images[0]

        # Dapatkan teks deskripsi dari style.json menggunakan style dan style2 sebagai key

        filename = f"Asset_{image_counter}.jpg"

        # Increment the counter for the next image
        image_counter += 1

        # Save the image with the generated filename
        image.save(os.path.join("D:/Inject Face Asset/Male", filename))

        # Save the image to a BytesIO buffer
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Convert the image to a base64-encoded string
        image_data = base64.b64encode(buffer.getvalue()).decode()

        # Return the HTML with the base64-encoded image
        return render_template("generate.html", image_data=image_data)
        # return jsonify({"image_base_64": image_data}), 200
    else:
        return render_template("generate.html")


@app.route("/static/style.json")
def get_style_json():
    with open("style.json") as f:
        style_data = json.load(f)
    return json.dumps(style_data)


@app.route("/static2/style2.json")
def get_style2_json():
    with open("style2.json") as f:
        style2_data = json.load(f)
    return json.dumps(style2_data)


@app.route("/static3/height.json")
def get_height_json():
    with open("height.json") as f:
        height_data = json.load(f)
    return json.dumps(height_data)


@app.route("/static4/width.json")
def get_width_json():
    with open("width.json") as f:
        width_data = json.load(f)
    return json.dumps(width_data)


if __name__ == "__main__":
    app.run(debug=True)
