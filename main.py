import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Betöltjük a környezeti változókat
load_dotenv()
CLARIFAI_API_KEY = os.getenv("CLARIFAI_API_KEY")

# FastAPI inicializálása
app = FastAPI()

# Pydantic modell a kérések validálására
class ImageRequest(BaseModel):
    image_url: str

# Clarifai API végpont
CLARIFAI_API_URL = "https://api.clarifai.com/v2/models/general-image-recognition/outputs"

@app.post("/clarifai/")
async def analyze_image(request: ImageRequest):
    """Elemzi a megadott képet a Clarifai API segítségével."""
    
    headers = {
        "Authorization": f"Key {CLARIFAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
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

    response = requests.post(CLARIFAI_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())

    return response.json()
