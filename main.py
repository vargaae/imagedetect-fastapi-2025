import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from typing import List

from dotenv import load_dotenv

load_dotenv()

   
PAT:str = ''
PAT:str = os.getenv("CLARIFAI_API_KEY")
if PAT is None:
    raise ValueError("CLARIFAI_API_KEY env key is unset!")
USER_ID:str = 'clarifai'
APP_ID:str = 'main'
MODEL_ID:str = 'general-image-recognition'
MODEL_VERSION_ID:str = 'aa7f35c01e0642fda5cf400f543e7c40'

app = FastAPI()

# 🔹 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clarifai API connection setup
channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)
metadata = (('authorization', 'Key ' + PAT),)

userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)


# Pydantic model for request input
class ImageInput(BaseModel):
    image_url: str


# Endpoint to process image using Clarifai model
@app.post("/predict/")
async def predict(image_input: ImageInput):
    # Prepare and make the Clarifai API call
    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            url=image_input.image_url
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )

    # Check for success response
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        raise HTTPException(status_code=500, detail=f"API call failed: {post_model_outputs_response.status.description}")

    # Process and return the predictions
    output = post_model_outputs_response.outputs[0]
    concepts = [{"name": concept.name, "value": concept.value} for concept in output.data.concepts]

    return {"predictions": concepts}
