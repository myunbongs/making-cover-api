from typing import Union
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

from .image_generate import Txt2img, Img2img, making_cover_stable_diffusion_txt2img, making_cover_stable_diffusion_img2img

from PIL import Image

app = FastAPI()

@app.post("/txt2img")
async def making_cover_txt2img(txt2img: Txt2img):
    global results
    results = making_cover_stable_diffusion_txt2img(txt2img)

    converted_results = jsonable_encoder(results)
    return JSONResponse(content=converted_results)

@app.post("/img2img")
async def making_cover_img2img(img2img: Img2img):
    global results
    results = making_cover_stable_diffusion_img2img(img2img)

    converted_results = jsonable_encoder(results)
    return JSONResponse(content=converted_results)