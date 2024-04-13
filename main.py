from diffusers import AutoPipelineForText2Image
import torch
import base64


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import random


import os
import io

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://cloud.redsols.us"
]

# cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)


pipe = None

@app.post("/generate")
async def generate(request: Request):
    global pipe
    data = await request.json()
    prompt = data["prompt"]
    
    if pipe is None:
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16")

    # PIL image
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        output_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # unique_filename = str(random.randint(0, 1000000000)) + ".png"

    # image.save(unique_filename)

    # output_base64 = None

    # with open(unique_filename, "rb") as buffer:
    #     output_base64 = base64.b64encode(buffer.read()).decode('utf-8')


    return JSONResponse(output_base64)


if __name__ == "__main__":
    import uvicorn

    if os.getenv("ENVIRONMENT") == "development":
        uvicorn.run("main:app", host="127.0.0.1", port=8000)
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=80)