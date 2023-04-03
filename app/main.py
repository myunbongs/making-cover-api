from typing import Union
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

from .image_generate import Txt2img, make_book_cover

from PIL import Image

app = FastAPI()

@app.post("/making-cover")
async def making_cover(txt2img: Txt2img):
    global results
    results = make_book_cover(txt2img)

    converted_results = jsonable_encoder(txt2img)
    return JSONResponse(content=converted_results)
