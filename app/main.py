from typing import Union
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

from .image_generate import Txt2img, Img2img, txt2img, img2img

from PIL import Image

app = FastAPI()

@app.post("/txt2img")
async def making_cover_txt2img(txt2img: Txt2img):
    global results
    results = txt2img(txt2img)

    converted_results = jsonable_encoder(txt2img)
    return JSONResponse(content=converted_results)

@app.post("/img2img")
async def making_cover_img2img(img2img: Img2img):
    global results
    results = img2img(img2img)

    converted_results = jsonable_encoder(img2img)
    return JSONResponse(content=converted_results)