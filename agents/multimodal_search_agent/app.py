import os

import streamlit as st
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from PIL import Image

import utils
from assistant import Assistant
from blip2 import generate_embeddings
from classifier import Classifier

load_dotenv("env/connection.env")
load_dotenv("env/llm.env")

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

model = ChatVertexAI(model_name=os.getenv("MODEL_NAME"), project=os.getenv("PROJECT_ID"), temperarture=0.0)
classifier = Classifier(model)
assistant = Assistant(model)

# Streamlit UI
st.title("Welcome to ZAAI's Fashion Assistant")

# User input section
user_input = st.text_input("Hi, I'm ZAAI's Fashion Assistant. How can I help you today?")

# User image input section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if st.button("Submit"):

    # understand what the user is asking for
    classification = classifier.classify(user_input)

    if uploaded_file:

        # Open the image file
        image = Image.open(uploaded_file)
        image.save("input_image.jpg")
        embedding = generate_embeddings(image=image)

    else:

        # create text embeddings in case the user does not upload an image
        embedding = generate_embeddings(text=user_input)

    # create a list of items to be retrieved and the path
    retrieved_items = []
    retrieved_items_path = []
    for item in classification.category:
        clothes = vector_db.similarity_search_by_vector(
            embedding, k=classification.number_of_items, filter={"category": {"$in": [item]}}
        )
        for clothe in clothes:
            retrieved_items.append({"bytesBase64Encoded": utils.encode_image_to_base64(clothe.page_content)})
            retrieved_items_path.append(clothe.page_content)

    # get assistant's recommendation
    assistant_output = assistant.get_advice(user_input, retrieved_items, len(retrieved_items))
    st.write(assistant_output.answer)

    cols = st.columns(len(retrieved_items)+1)
    for col, retrieved_item in zip(cols, ["input_image.jpg"]+retrieved_items_path):
        # Display the image in the respective column
        col.image(retrieved_item)

    user_input = st.text_input("")

else:
    st.warning("Please provide text.")
