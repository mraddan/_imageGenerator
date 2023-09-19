from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    PNDMScheduler,
KDPM2DiscreteScheduler
)
import torch
import cv2
import json

logo = "logo-wizard/logo-diffusion-checkpoint"
refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"

l_sch = PNDMScheduler.from_pretrained(
    logo,
    subfolder="scheduler")
r_sch = EulerAncestralDiscreteScheduler.from_pretrained(
    refiner,
    subfolder="scheduler")

l_pipe = DiffusionPipeline.from_pretrained(
    logo,
    torch_dtype=torch.float16,
    scheduler=l_sch,
)
r_pipe = DiffusionPipeline.from_pretrained(
    refiner,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    scheduler=r_sch,
    vae=l_pipe.vae,
)
r_pipe.enable_model_cpu_offload()
l_pipe.enable_model_cpu_offload()
r_pipe.enable_attention_slicing()
l_pipe.enable_attention_slicing()
r_pipe.enable_vae_slicing()
l_pipe.enable_vae_slicing()
r_pipe.enable_xformers_memory_efficient_attention()
l_pipe.enable_xformers_memory_efficient_attention()

logo_np = "low quality, worst quality, bad composition, extra digit, fewer digits, inscription, asymmetric, ugly, tiling, out of frame, \
    disfigured, deformed, body out of frame, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur"
with open("stylelogo.json") as file:
    main_style = json.load(file)
    file.close()

output_count = 76
prompt_history_file = "prompt.txt"

while True:
    prompt = str(input('Masukan Prompt: '))
    if prompt =="":
        break
    inp_style1 = input("Masukan Style: ")
    m_style = main_style.get(inp_style1)


    with open(prompt_history_file, "a") as prompt_file:
        prompt_file.write(f'{output_count}. {prompt} ({inp_style1}) \n')

    image_gen = l_pipe(
            prompt=prompt + m_style ,
            negative_prompt=logo_np,
            num_inference_steps=35,
            height=768,
            width=768,
            guidance_scale=7,
        ).images[0]

    
        # Simpan gambar dengan penamaan angka berurut
    output_filename = f"{output_count:02d}.jpg"
    image_gen.save(output_filename)
    
    print(f"Gambar disimpan: {output_filename}")
    
    # Increment angka urutan
    output_count += 1

    