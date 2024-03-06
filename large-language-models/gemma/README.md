# Gemma

**Steps to run the code:**
1. Install docker
2. Create a virtual environment with python 3.10.13
`conda create --name myenv python=3.10.13`
3. Activate in your new virtual environment
`conda activate myenv`
4. Install the required requirements
`pip install -r requirements.txt`
5. Create a folder called `/data` under `gemma/` and add review data from https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset
6. Create a folder called `/env` under `gemma/` and add a file with the following:
    - postgres.env
    ```
    POSTGRES_DB=postgres
    POSTGRES_USER=admin
    POSTGRES_PASSWORD=root
    ```
    - connection.env
    ```
    DRIVER=psycopg2
    HOST=postgres
    PORT=5432     
    DATABASE=postgres
    USERNAME=admin
    PASSWORD=root
    ```
7. Download Mistral 7B model `mistral-7b-instruct-v0.1.Q4_K_M.gguf` from https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main and add it to `gemma/model/`
8. Download Llama 7B model `nous-hermes-llama-2-7b.Q4_K_M.gguf` from https://huggingface.co/TheBloke/Nous-Hermes-Llama-2-7B-GGUF and add it to `gemma/model/`
9. Download Gemma 7B model `gemma-7b-it-Q4_K_M.gguf` from https://huggingface.co/rahuldshetty/gemma-7b-it-gguf-quantized/tree/main and add it to `gemma/model/`
10. Run the command `docker-compose up --build`
11. Run in the notebook `gemma.ipynb` 

## Folder Structure:
------------

    ├── gemma
    │
    ├────────── base                                          <- Configuration class
    ├────────── encoder                                       <- Encoder class
    ├────────── generator                                     <- Generator class
    ├────────── retriever                                     <- Retriever class
    ├────────── data                                          <- csv file
    ├────────── env                                           <- env files
    ├────────── model                                         <- GGUF models
    │
    │────────── config.yaml                                   <- Config definition
    │
    │────────── gemma.ipynb                                   <- notebook
    │
    │──────── requirements.txt                                <- package versions
    └──────── docker-compose.yaml
--------
