# Retrieval Augmented Generation (RAG)

**Steps to run the code:**
1. Create a virtual environment with python 3.10.13
`conda create --name myenv python=3.10.13`
2. Activate in your new virtual environment
`conda activate myenv`
3. Install the required requirements
`pip install -r requirements.txt`
4. Create a folder called model under `rag/`
5. Download Llama model `nous-hermes-llama-2-7b.Q4_0.gguf` from https://huggingface.co/TheBloke/Nous-Hermes-Llama-2-7B-GGUF/tree/main and add it to `model/`
5. Run the notebook

## Folder Structure:
------------

    ├── RAG
    │
    ├──────── base           <- Configuration class
    ├──────── encoder        <- Encoder class
    ├──────── generator      <- Generator class
    ├──────── retriever      <- Retriever class
    │
    │──── config.yaml        <- Config definition
    │──── requirements.txt   <- package version for installing
    │
    └──── rag.ipynb          <- notebook to run the code
--------
