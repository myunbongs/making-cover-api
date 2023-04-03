from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import random
from typing import Optional
from fastapi import HTTPException

from .utils import image2string

import io

class Txt2img(BaseModel):
    model: Optional[str]
    prompt: Optional[str] 
    negative_prompt: Optional[str] 
    height: Optional[int] 
    width: Optional[int] 
    number_of_imgs: Optional[int] 
    imgs : Optional[str] = []
    seeds : Optional[int] = []

def make_book_cover(txt2img: Txt2img):

    model = select_model(txt2img.model)

    generator = torch.Generator(device="cuda")
    seed = generator.seed()
    txt2img.seeds.append(seed)
    generator = generator.manual_seed(seed)
    
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16).to(
        "cuda"
    )
    
    pipe.enable_xformers_memory_efficient_attention()

    with torch.inference_mode():
        imgs = pipe(prompt=txt2img.prompt, negative_prompt=txt2img.negative_prompt, \
                height=txt2img.height, width=txt2img.width, num_inference_steps=30, \
                num_images_per_prompt=txt2img.number_of_imgs, guidance_scale=7.5, generator=generator).images        
   
        for i, img in enumerate(imgs):
            file_name = "./app/images/{}_{}_{}.png".format(txt2img.prompt, generator.seed, i)
            img.save(file_name)
            txt2img.imgs.append(image2string(img))

    return txt2img

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
