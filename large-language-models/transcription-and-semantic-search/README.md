# Multilingual Transcription and Semantic Search

**Steps to run the code:**
1. Install docker
2. Create a virtual environment with python 3.10.13
`conda create --name myenv python=3.10.13`
3. Activate in your new virtual environment
`conda activate myenv`
4. Install the required requirements
`pip install -r requirements.txt`
5. Create a folder called `/data` under `transcription-and-semantic-search/` and add your videos
6. Create a folder called `/env` and add 3 files with the following:
    - connection.env
    ```
    DRIVER=psycopg2
    HOST=localhost
    PORT=5432     
    DATABASE=postgres
    USERNAME=admin
    PASSWORD=root
    ```
    - pgadmin.env
    ```
    PGADMIN_DEFAULT_EMAIL=admin@admin.com
    PGADMIN_DEFAULT_PASSWORD=root
    ```
    - postgres.env
    ```
    POSTGRES_DB=postgres
    POSTGRES_USER=admin
    POSTGRES_PASSWORD=root
    ```
6. Run the command `docker-compose up -d`
7. Run the notebook

## Folder Structure:
------------

    ├── transcription-and-semantic-search
    │
    ├──────── base                                          <- Configuration class
    ├──────── encoder                                       <- Encoder class
    ├──────── transcriptor                                  <- WhisperX class
    ├──────── data                                          <- videos and audios
    ├──────── env                                           <- env files
    │
    │──── config.yaml                                       <- Config definition
    │──── requirements.txt                                  <- package version for installing
    │
    └──── multilingual_transcription_semantic_search.ipynb  <- notebook to run the code
--------
