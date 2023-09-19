from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    DDIMParallelScheduler
)
# import torch
import cv2
import json

model = "stabilityai/stable-diffusion-xl-base-1.0"
refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"

m_sch = DDIMParallelScheduler.from_pretrained(
    model,
    subfolder="scheduler",)
r_sch = EulerAncestralDiscreteScheduler.from_pretrained(
    refiner,
    subfolder="scheduler",)
    
m_pipe = DiffusionPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
    variant="fp16",
    scheduler=m_sch,)
r_pipe = DiffusionPipeline.from_pretrained(
    refiner,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    scheduler=r_sch,
    vae=m_pipe.vae,
    text_encoder_2=m_pipe.text_encoder_2,)

m_pipe.enable_model_cpu_offload()
r_pipe.enable_model_cpu_offload()
m_pipe.enable_attention_slicing()
r_pipe.enable_attention_slicing()
m_pipe.enable_vae_slicing()
r_pipe.enable_vae_slicing()
m_pipe.enable_xformers_memory_efficient_attention()
r_pipe.enable_xformers_memory_efficient_attention()

image_np = "double face, hands, wrist, Ugly, Duplicate, Extra fingers, Mutated hands, Poorly drawn face, Mutation, Deformed, \
    Blurry, Bad anatomy, Bad proportions, Extra limbs, cloned face, Disfigured, Missing arms, Missing legs, Extra arms, \
    Extra legs, Fused fingers, Too many fingers, Long neck, writing, letters, Multiple bodies, multiple heads, extra hands, \
    extra fingers, ugly, skinny, extra leg, extra foot, blur, bad anatomy, double body, stacked body, fused hands, fused body, \
    fused heads, fused legs, fused feet, multiple faces, ((conjoined)), (siamese twin), double faces, two faces, texts, \
    watermarked, watermark, logo, face out of frame, stacked background, ((out of frame portrait)), bucktoothed, cropped"

with open("style.json") as file:
    main_style = json.load(file)
    file.close()
with open("style2.json") as file:
    opt_style = json.load(file)
    file.close()

output_folder = "D:/image/Sample Pictures/Sample Pictures"  # Ubah sesuai dengan folder tujuan penyimpanan gambar
output_count = 25


prompt_history_file = "prompt.txt"

while True:
    prompt = str(input('Masukan Prompt: '))
    if prompt =="":
        break
    inp_style1 = input("Masukan Style1: ")
    m_style = main_style.get(inp_style1)
    inp_style2 = input("Masukan Style2: ")
    o_style = opt_style.get(inp_style2)

    with open(prompt_history_file, "a") as prompt_file:
        prompt_file.write(f'{output_count}. {prompt} ({inp_style1},{inp_style2}) "\n"')

    image_gen = m_pipe(
            prompt=prompt + m_style + o_style,
            negative_prompt=image_np,
            num_inference_steps=35,
            height=768,
            width=768,
            guidance_scale=7,
            output_type="latent",
        ).images

    image_final = r_pipe(
            prompt=prompt + m_style + o_style,
            negative_prompt=image_np,
            num_inference_steps=35,
            guidance_scale=7,
            image=image_gen,
        ).images[0]
    
        # Simpan gambar dengan penamaan angka berurut
    output_filename = f"{output_count:02d}.jpg"
    image_final.save(output_filename)
    
    print(f"Gambar disimpan: {output_filename}")
    
    # Increment angka urutan
    output_count += 1

    
