from typing import Optional
from PIL import Image

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoProcessor, Blip2TextModelWithProjection, Blip2VisionModelWithProjection

PROCESSOR = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")
TEXT_MODEL = Blip2TextModelWithProjection.from_pretrained("Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16).to("cpu")
IMAGE_MODEL = Blip2VisionModelWithProjection.from_pretrained("Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16).to("cpu")

image = Image.open('data/jersey/10364.jpg')

def generate_embeddings(text: Optional[str] = None, image: Optional[Image] = None) -> np.ndarray:
    """
    Generate embeddings from text or image using the Blip2 model.
    Args:
        text (Optional[str]): customer input text
        image (Optional[Image]): customer input image
    Returns:
        np.ndarray: embedding vector
    """
    if text:
        inputs = PROCESSOR(text=text, return_tensors="pt").to("cpu")
        outputs = TEXT_MODEL(**inputs)
        embedding = F.normalize(outputs.text_embeds, p=2, dim=1)[:, 0, :].detach().numpy().flatten()
    else:
        inputs = PROCESSOR(images=image, return_tensors="pt").to("cpu", torch.float16)
        outputs = IMAGE_MODEL(**inputs)
        embedding = F.normalize(outputs.image_embeds, p=2, dim=1).mean(dim=1).detach().numpy().flatten()

    return embedding

from langchain_postgres.vectorstores import PGVector
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# GCP Vertex AI setup
PROJECT_ID = "ms-selfaudit-dev"
LOCATION = "us-central1"
MODEL_NAME = "gemini-1.5-pro-002"

# PostgreSQL configuration
DB_CONFIG = {
    "dbname": "selfaudit",
    "user": "admin",
    "password": "fV(\u003eu1$Nt97EU9O}CVXnv=7IlJn|.z-r",
    "host": "34.174.156.67",
    "port": "5432",
    "table_name": "multimodal_search",
}
connection = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
collection_name = "fashion"

vector_store = PGVector(
    embeddings=HuggingFaceEmbeddings(model_name='nomic-ai/modernbert-embed-base'), # does not matter for our use case
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
