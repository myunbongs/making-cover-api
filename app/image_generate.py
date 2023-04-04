from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import random
from typing import Optional
from fastapi import HTTPException
import os 

from .utils import image2string

import io

class Txt2img(BaseModel):
    model: Optional[str]
    prompt: Optional[str] 
    negative_prompt: Optional[str] 
    height: Optional[int] 
    width: Optional[int] 
    num_inference_steps: Optional[int] = 30 
    guidance_scale: Optional[int] = 7.5 
    number_of_imgs: Optional[int] = 1 
    imgs : Optional[str] = []
    seeds : Optional[int] = []

class Img2img(BaseModel): 
    model: Optional[str]
    init_image_num: Optional[int]
    prompt: Optional[str] 
    negative_prompt: Optional[str] 
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[int] = 7.5 
    strength: Optional[float] = 0.8
    number_of_imgs: Optional[int] = 1
    imgs : Optional[str] = []
    seeds : Optional[int] = []

def making_cover_stable_diffusion_txt2img(txt2img: Txt2img):

    model = select_model(txt2img.model)

    generator = [torch.Generator(device="cuda").manual_seed(0) for i in range(txt2img.number_of_imgs)]
    img2img.seeds = generator

    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16).to(
        "cuda"
    )
    
    pipe.enable_xformers_memory_efficient_attention()

    with torch.inference_mode():
        imgs = pipe(prompt=txt2img.prompt, negative_prompt=txt2img.negative_prompt, 
                height=txt2img.height, width=txt2img.width, num_inference_steps=txt2img.num_inference_steps, 
                num_images_per_prompt=txt2img.number_of_imgs, guidance_scale=txt2img.guidance_scale, generator=generator).images        
   
        for i, img in enumerate(imgs):
            file_name = "./app/images/{}_{}_{}.png".format(txt2img.prompt, generator.seed, i)
            img.save(file_name)
            txt2img.imgs.append(image2string(img))

    return txt2img

def making_cover_stable_diffusion_img2img(img2img: Img2img):

    total_init_imgs_num = len(os.listdir('./app/init_images/'))
    init_img = select_init_image(img2img.init_image_num, total_init_imgs_num)

    model = select_model(img2img.model)

    generator = [torch.Generator(device="cuda").manual_seed(0) for i in range(img2img.number_of_imgs)]
    img2img.seeds = generator

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model, torch_dtype=torch.float16).to(
        "cuda"
    )
    
    pipe.enable_xformers_memory_efficient_attention()

    with torch.inference_mode():
        imgs = pipe(prompt=img2img.prompt, negative_prompt=img2img.negative_prompt,
                image=init_img, num_inference_steps=img2img.num_inference_steps, 
                num_images_per_prompt=img2img.number_of_imgs, strength = img2img.strength, guidance_scale=img2img.guidance_scale, 
                generator=generator).images   
   
        for i, img in enumerate(imgs):
            file_name = "./app/images/{}_{}_{}.png".format(img2img.prompt, generator.seed, i)
            img.save(file_name)
            img2img.imgs.append(image2string(img))

    return img2img


def select_model(model_name):
    if model_name == "stable-diffusion":
        model = "runwayml/stable-diffusion-v1-5"
    elif model_name == "anything":
        model = "andite/anything-v4.0"
    elif model_name == "pastelmix":
        model = "andite/pastel-mix"
    else:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

def select_init_image(init_image_num, total_num):
    for i in range(total_num):
        if init_image_num == i:
            init_image = Image.open('./app/init_images/{}.jpg'.format(i)).convert("RGB")
    return init_image