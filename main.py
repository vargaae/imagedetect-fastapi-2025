from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# 🔹 CORS engedélyezése
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- MEGENGEDETT ORIGINEK
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Clarifai API konfiguráció
CLARIFAI_API_KEY = os.getenv("CLARIFAI_API_KEY")
CLARIFAI_URL = "https://api.clarifai.com/v2/models/general-image-recognition/outputs"

@app.post("/clarifai/")
async def analyze_image(data: dict):
    headers = {
        "Authorization": f"Key {CLARIFAI_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(CLARIFAI_URL, json={
        "inputs": [{"data": {"image": {"url": data["image_url"]}}}]
    }, headers=headers)
    
    return response.json()
