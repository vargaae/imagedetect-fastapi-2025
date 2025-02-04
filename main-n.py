from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# 🔹 Környezeti változók (vagy írd be közvetlenül)
CLARIFAI_API_KEY = os.getenv("CLARIFAI_API_KEY", "YOUR_CLARIFAI_API_KEY")
CLARIFAI_USER_ID = "clarifai"
CLARIFAI_APP_ID = "main"
CLARIFAI_MODEL_ID = "general-image-recognition"
CLARIFAI_MODEL_VERSION_ID = 'aa7f35c01e0642fda5cf400f543e7c40'
# Change these to whatever model and image URL you want to use


# 🔹 CORS engedélyezése
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- MEGENGEDETT ORIGINEK
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Request body modell
class ImageRequest(BaseModel):
    image_url: str

@app.post("/clarifai")
def analyze_image(request: ImageRequest):
    headers = {
        "Authorization": f"Key {CLARIFAI_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "user_app_id": {
            "user_id": CLARIFAI_USER_ID,
            "app_id": CLARIFAI_APP_ID
        },
        "inputs": [
            {
                "data": {
                    "image": {
                        "url": request.image_url
                    }
                }
            }
        ]
    }

    clarifai_url = f"https://api.clarifai.com/v2/models/{CLARIFAI_MODEL_ID}/outputs"

    response = requests.post(clarifai_url, headers=headers, json=data)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())

    return response.json()
