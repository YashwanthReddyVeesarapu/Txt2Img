from diffusers import AutoPipelineForText2Image
import torch
import base64


from fastapi import FastAPI, Request
from fastapi.responses import Response

import os


app = FastAPI()

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

    # convert to base64
    image_bytes = image.tobytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    response_base64 = "data:image/png;base64," + image_base64

    return Response(content=response_base64, media_type="text/plain")

if __name__ == "__main__":
    import uvicorn

    if os.getenv("ENVIRONMENT") == "development":
        uvicorn.run("main:app", host="127.0.0.1", port=8000)
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=443)