# Translation

**Steps to run the code:**
1. Install docker
2. Create a virtual environment with python 3.10.13
`conda create --name myenv python=3.10.13`
3. Activate in your new virtual environment
`conda activate myenv`
4. Install the required requirements
`pip install -r requirements.txt`
5. Create a folder called `/data` under `translation/src/` and add review data from https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset
6. Create a folder called `/env` under `translation/src/` and add a file with the following:
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
7. Download Llama model `nous-hermes-llama-2-7b.Q4_0.gguf` from https://huggingface.co/TheBloke/Nous-Hermes-Llama-2-7B-GGUF/tree/main and add it to `translation/src/model/`
8. Run the command `docker-compose up --build` and you can open http://localhost:8501/ in your browser and chat!
9. Or you can run in the notebook `translation.ipynb` 

## Folder Structure:
------------

    ├── translation
    │
    ├──────── src 
    ├────────── base                                          <- Configuration class
    ├────────── classifier                                    <- Language Detector class
    ├────────── encoder                                       <- Encoder class
    ├────────── generator                                     <- Generator class
    ├────────── retriever                                     <- Retriever class
    ├────────── translator                                    <- Translator class
    ├────────── data                                          <- csv file
    ├────────── env                                           <- env files
    ├────────── model                                         <- GGUF Llama
    │
    │────────── config.yaml                                   <- Config definition
    │────────── lang_map.yaml                                 <- language mapping between XLM-RoBERTa and mBART
    │
    │────────── translation.ipynb                             <- notebook
    │────────── populate.py                                   <- python script to populate PGVector
    │────────── app.py                                        <- streamlit application to chat with our LLM
    │
    │──────── requirements.txt                                <- package versions
    │──────── docker-compose.yaml
    └──────── Dockerfile
--------
