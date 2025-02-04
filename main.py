from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from typing import List

from newsapi import NewsApiClient


# Define your variables
PAT = '8223810da7484e638622d62d141fc442'
USER_ID = 'clarifai'
APP_ID = 'main'
MODEL_ID = 'general-image-recognition'
MODEL_VERSION_ID = 'aa7f35c01e0642fda5cf400f543e7c40'

# Initialize FastAPI app
app = FastAPI()

# 🔹 CORS engedélyezése
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- MEGENGEDETT ORIGINEK
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

# # Initialize the NewsApiClient
# api = NewsApiClient(api_key='c6b7a51f201b426c9cfe9c5053d9bd5f')

# # Pydantic models for returning articles
# class Article(BaseModel):
#     source: dict
#     author: str
#     title: str
#     description: str
#     url: str
#     publishedAt: str
#     content: str

# # Fetch top headlines
# @app.get("/top_headlines", response_model=List[Article])
# async def get_top_headlines():
#     try:
#         # Fetch top headlines for 'chatgpt' from selected sources
#         top_headlines = api.get_top_headlines(
#             q='chatgpt',
#             sources='bbc-news,the-verge',
#             category='science',
#             language='en',
#             country='us'
#         )

#         # Check if articles are available in response
#         if top_headlines['status'] != 'ok':
#             raise HTTPException(status_code=500, detail="Failed to fetch top headlines")

#         return top_headlines['articles']

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error fetching top headlines: {e}")


# # Fetch all articles
# @app.get("/all_articles", response_model=List[Article])
# async def get_all_articles():
#     try:
#         # Fetch all articles for 'chatgpt' from selected sources, domains, and date range
#         all_articles = api.get_everything(
#             q='chatgpt',
#             sources='bbc-news,the-verge',
#             domains='bbc.co.uk,techcrunch.com',
#             from_param='2025-01-01',
#             to='2025-02-04',
#             language='en',
#             sort_by='relevancy',
#             page=2
#         )

#         # Check if articles are available in response
#         if all_articles['status'] != 'ok':
#             raise HTTPException(status_code=500, detail="Failed to fetch all articles")

#         return all_articles['articles']

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error fetching all articles: {e}")