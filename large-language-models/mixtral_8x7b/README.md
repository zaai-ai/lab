# Mistral

**Steps to run the code:**
1. Install docker
2. Create a virtual environment with python 3.10.13
`conda create --name myenv python=3.10.13`
3. Activate in your new virtual environment
`conda activate myenv`
4. Install the required requirements
`pip install -r requirements.txt`
5. Create a folder called `/data` under `mixtral_8x7b/` and add review data from https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset
6. Create a folder called `/env` under `mixtral_8x7b/` and add a file with the following:
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
7. Download Mistral 7B model `mistral-7b-v0.1.Q4_K_M.gguf` from https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF and add it to `mixtral_8x7b/model/`
8. Download Llama 7B model `nous-hermes-llama-2-7b.Q4_K_M.gguf` from https://huggingface.co/TheBloke/Nous-Hermes-Llama-2-7B-GGUF and add it to `mixtral_8x7b/model/`
9. Download Mixtral 8x7B model `mixtral-8x7b-v0.1.Q4_K_M.gguf` from https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF and add it to `mixtral_8x7b/model/`
10. Download Llama 70B model `llama-2-70b-chat.Q4_K_M.gguf` from https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF and add it to `mixtral_8x7b/model/`
11. Run the command `docker-compose up --build`
12. Run in the notebook `mixtral_8x7b.ipynb` 

## Folder Structure:
------------

    ├── mixtral_8x7b
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
    │────────── mixtral_8x7b.ipynb                            <- notebook
    │
    │──────── requirements.txt                                <- package versions
    └──────── docker-compose.yaml
--------
