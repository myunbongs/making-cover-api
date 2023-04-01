from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import random
from typing import Optional
from fastapi import HTTPException

from utils import image2string

import io

class Cover(BaseModel):
    model: Optional[str]
    prompt: Optional[str] 
    negative_prompt: Optional[str] 
    height: Optional[int] 
    width: Optional[int] 
    number_of_imgs: Optional[int] 
    imgs : Optional[str] = []
    seed : Optional[int]

def make_book_cover(cover: Cover):

    if cover.model == "stable-diffusion":
        model = "runwayml/stable-diffusion-v1-5"
    elif cover.model == "anything":
        model = "andite/anything-v4.0"
    elif cover.model == "pastelmix":
        model = "andite/pastel-mix"
    else:
        raise HTTPException(status_code=404, detail="Model not found")

    seed = random.randint(1, 2147483647)
    generator = torch.Generator("cuda").manual_seed(seed)
    
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16).to(
        "cuda"
    )

    pipe.enable_xformers_memory_efficient_attention()
    
    with torch.inference_mode():
        imgs = pipe(prompt=cover.prompt, negative_prompt=cover.negative_prompt, \
                height=cover.height, width=cover.width, num_inference_steps=30, \
                num_images_per_prompt=cover.number_of_imgs, guidance_scale=7.5, generator=generator).images        
   
        for i, img in enumerate(imgs):
            file_name = "./images/{}_{}_{}.png".format(cover.prompt, cover.seed, i)
            img.save(file_name)
            cover.imgs.append(image2string(img))

        cover.seed = seed

    return cover
