from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    PNDMScheduler,
)
import torch
import cv2
import numpy as np
from PIL import Image

import gfpgan
import insightface
from insightface.app import FaceAnalysis

from flask import Flask, request, render_template, abort

from io import BytesIO
import gdown
import json
import gc
import os.path
import uuid
import base64

app = Flask(__name__)

# Face Model
face = FaceAnalysis(name='buffalo_l', root=".")
swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx')
face.prepare(ctx_id=0, det_size=(640, 640))

# Model Loading
model = "stabilityai/stable-diffusion-xl-base-1.0"
refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"
logo = "logo-wizard/logo-diffusion-checkpoint"

enhancer = gfpgan.GFPGANer(model_path="models/GFPGANv1.4.pth", upscale=1)

m_sch = PNDMScheduler.from_pretrained(
    model,
    subfolder="scheduler",
    cache_dir="./refiners",
)
r_sch = EulerAncestralDiscreteScheduler.from_pretrained(
    refiner,
    subfolder="scheduler",
    cache_dir="./refiners",
)
l_sch = EulerAncestralDiscreteScheduler.from_pretrained(
    logo,
    subfolder="scheduler",
    cache_dir="./refiners",
)

m_pipe = DiffusionPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
    cache_dir="models/",
    variant="fp16",
    scheduler=m_sch,
)
r_pipe = DiffusionPipeline.from_pretrained(
    refiner,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    cache_dir="models/",
    scheduler=r_sch,
    vae=m_pipe.vae,
    text_encoder_2=m_pipe.text_encoder_2,
)
l_pipe = DiffusionPipeline.from_pretrained(
    logo,
    torch_dtype=torch.float16,
    cache_dir="models/",
    scheduler=l_sch,
)

# Tweaking model
m_pipe.enable_model_cpu_offload()
r_pipe.enable_model_cpu_offload()
l_pipe.enable_model_cpu_offload()

m_pipe.enable_attention_slicing()
r_pipe.enable_attention_slicing()
l_pipe.enable_attention_slicing()

m_pipe.enable_vae_slicing()
r_pipe.enable_vae_slicing()
l_pipe.enable_vae_slicing()

m_pipe.enable_xformers_memory_efficient_attention()
r_pipe.enable_xformers_memory_efficient_attention()
l_pipe.enable_xformers_memory_efficient_attention()

# Parameter
i_steps = 40
l_steps = 20

image_np = "double face, hands, wrist, Ugly, Duplicate, Extra fingers, Mutated hands, Poorly drawn face, Mutation, Deformed, \
    Blurry, Bad anatomy, Bad proportions, Extra limbs, cloned face, Disfigured, Missing arms, Missing legs, Extra arms, \
    Extra legs, Fused fingers, Too many fingers, Long neck, writing, letters, Multiple bodies, multiple heads, extra hands, \
    extra fingers, ugly, skinny, extra leg, extra foot, blur, bad anatomy, double body, stacked body, fused hands, fused body, \
    fused heads, fused legs, fused feet, multiple faces, ((conjoined)), (siamese twin), double faces, two faces, texts, \
    watermarked, watermark, logo, face out of frame, stacked background, ((out of frame portrait)), bucktoothed, cropped"

logo_np = "low quality, worst quality, bad composition, extra digit, fewer digits, inscription, asymmetric, ugly, tiling, out of frame, \
    disfigured, deformed, body out of frame, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur"

# Style
with open("assets/generation/main_style.json") as file:
    main_style = json.load(file)
    file.close()
with open("assets/generation/opt_style.json") as file:
    opt_style = json.load(file)
    file.close()
with open("assets/logo/style.json") as file:
    logo_style = json.load(file)
    file.close()


# Helper
def convert_image(image):
    # Convert Image To Base 64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")

    return {"image": base64.b64encode(buffered.getvalue()).decode()}


def swap_face(image, path):
    im_style = cv2.imread(path)

    user = face.get(image)
    swap = face.get(im_style)

    result = swapper.get(im_style, swap[0], user[0], paste_back=True)

    return result


def enhance_face(image):
    _, _, result = enhancer.enhance(image, paste_back=True)

    return Image.fromarray(result[:, :, ::-1])


def rename_file(file, gender):
    if gender:
        os.rename(file, f"assets/avatar/male/{uuid.uuid4()}.jpg")
    else:
        os.rename(file, f"assets/avatar/female/{uuid.uuid4()}.jpg")


# API Method
@app.get("/")
def index():
    return """Access image by POST /generate/image\n
            Access avatar by POST /generate/avatar\n
            Access logo by POST /generate/logo"""


@app.post("/generate/image")
def image():
    data = request.json

    try:
        prompt = data["prompt"]
        m_style = main_style[data["main_style"]]
        o_style = opt_style[data["opt_style"]]
        size = int(data["size"])
        refiner = data["refiner"]
    except KeyError:
        return "Missing Key in Request Data"
    except ValueError:
        return "Invalid Data in Request Data"

    if refiner == 1:
        image_gen = m_pipe(
            prompt=prompt + m_style + o_style,
            negative_prompt=image_np,
            num_inference_steps=i_steps,
            height=size,
            width=size,
            guidance_scale=7,
            output_type="latent",
        ).images

        image_final = r_pipe(
            prompt=prompt + m_style + o_style,
            negative_prompt=image_np,
            num_inference_steps=i_steps,
            guidance_scale=7,
            image=image_gen,
        ).images[0]
    else:
        image_final = m_pipe(
            prompt=prompt + m_style + o_style,
            negative_prompt=image_np,
            num_inference_steps=i_steps,
            height=size,
            width=size,
            guidance_scale=7,
        ).images[0]

    data = convert_image(image_final)

    response = app.response_class(
        response=json.dumps(data), status=200, mimetype="application/assets"
    )

    gc.collect()

    return response


@app.post("/generate/logo")
def logo():
    data = request.json

    try:
        prompt = data["prompt"]
        l_style = logo_style[data["l_style"]]
        size = int(data["size"])
        refiner = data["refiner"]
    except:
        return "Missing Data In Requests"

    if refiner == 1:
        image_log = l_pipe(
            prompt=prompt + l_style,
            negative_prompt=logo_np,
            num_inference_steps=i_steps,
            height=size,
            width=size,
            guidance_scale=7.5,
            output_type="latent",
        ).images

        image_final = refiner(
            prompt=prompt + l_style,
            negative_prompt=logo_np,
            num_inference_steps=l_steps,
            image=image_log,
        ).images[0]
    else:
        image_final = l_pipe(
            prompt=prompt + l_style,
            negative_prompt=logo_np,
            num_inference_steps=i_steps,
            height=size,
            width=size,
            guidance_scale=7.5,
        ).images[0]

    data = convert_image(image_final)

    response = app.response_class(
        response=json.dumps(data), status=200, mimetype="application/assets"
    )

    gc.collect()

    return response


@app.post("/generate/avatar")
def avatar():
    path = request.form['path']

    image_file = request.files['image']
    if image_file.filename == '':
        return "No selected file"

    # Read Image
    image_stream = image_file.read()
    image_array = np.frombuffer(image_stream, np.uint8)

    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    img = swap_face(img, path)
    img = enhance_face(img)

    data = convert_image(img)

    response = app.response_class(
        response=json.dumps(data), status=200, mimetype="application/assets"
    )

    gc.collect()

    return response


@app.get("/male_assets")
def get_male():
    abs_path = "/home/trippy/ImageGeneration-Generator/assets/avatar/male"

    if not os.path.exists(abs_path):
        return abort(404)

    files = os.listdir(abs_path)
    return render_template('generate.html', files=files)


@app.get("/female_assets")
def get_female():
    abs_path = "/home/trippy/ImageGeneration-Generator/assets/avatar/female"

    if not os.path.exists(abs_path):
        return abort(404)

    files = os.listdir(abs_path)
    return render_template('generate.html', files=files)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
