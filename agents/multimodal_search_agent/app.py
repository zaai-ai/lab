import streamlit as st
from assistant import Assistant
from classifier import Classifier
from blip2 import generate_embeddings
from PIL import Image
from langchain_google_vertexai import ChatVertexAI
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
    embeddings=HuggingFaceEmbeddings('nomic-ai/modernbert-embed-base'), # does not matter for our use case
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

model = ChatVertexAI(model_name=MODEL_NAME, project=PROJECT_ID, temperarture=0.0)
classifier = Classifier(model)
assistant = Assistant(model)

# Streamlit UI
st.title("Multimodal Chat App with Vertex AI and BLIP-2")

# User input section
user_input = st.text_input("Enter text:")

if st.button("Submit"):
    if user_input:

        classification = classifier.classify(user_input)
        st.write(f"The customer is looking: {classification.category}")

        embedding = generate_embeddings(user_input)
        retrieved_items = {}
        for item in classification.category:
            retrieved_items[item] = vector_store.similarity_search_by_vector(embedding, k=1, filter={"category": {"$in": [item]}})

        st.subheader("Top Retrieved Items:")
        assistant_output = assistant.get_advice(user_input, retrieved_items)
        st.write(assistant_output.answer)

    else:
        st.warning("Please provide text or an image.")