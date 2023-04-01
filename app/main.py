from typing import Union
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

from image_generate import Cover, make_book_cover

from PIL import Image

app = FastAPI()

@app.post("/making-cover")
async def making_cover(cover: Cover):
    global results
    results = make_book_cover(cover)

    converted_results = jsonable_encoder(cover)
    return JSONResponse(content=converted_results)
