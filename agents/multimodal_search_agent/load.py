import glob
import os

from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from PIL import Image

from blip2 import generate_embeddings

load_dotenv("env/connection.env")

# PGVector configuration
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.getenv("DRIVER"),
    host=os.getenv("HOST"),
    port=os.getenv("PORT"),
    database=os.getenv("DATABASE"),
    user=os.getenv("USERNAME"),
    password=os.getenv("PASSWORD"),
)

# Connnect to vector db
vector_db = PGVector(
    embeddings=HuggingFaceEmbeddings(model_name="nomic-ai/modernbert-embed-base"),  # does not matter for our use case
    collection_name="fashion",
    connection=CONNECTION_STRING,
    use_jsonb=True,
)


if __name__ == "__main__":

    # generate image embeddings
    # save path to image in text
    # save category in metadata
    texts = []
    embeddings = []
    metadatas = []

    for category in glob.glob("images/*"):
        cat = category.split("/")[-1]
        for img in glob.glob(f"{category}/*"):
            texts.append(img)
            embeddings.append(generate_embeddings(image=Image.open(img)).tolist())
            metadatas.append({"category": cat})

    vector_db.add_embeddings(texts, embeddings, metadatas)
