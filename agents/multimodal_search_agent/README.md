# Multi Modal Search

**Steps to run the code:**
1. Create a virtual environment with python 3.10.15
`conda create --name myenv python=3.10.15`
2. Activate in your new virtual environment
`conda activate myenv`
3. Install the required requirements
    - `pip install -r requirements.txt`
4. Create a folder called `/env` and add 4 files with the following:
    - connection.env
    ```
    DRIVER=psycopg
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
    - llm.env
    ```
    MODEL_NAME=<model name>
    PROJECT_ID=<gcp project>
    ```
5. Run the command `docker-compose up -d`
5. Run the load.py file
6. Run the command `streamlit run app.py`

## Folder Structure:
------------

    ├── multimodal_search_agent
    │
    ├────── images                      <- images to be loaded to vector db
    ├────── env                         <- agent and vector db configuration
    │
    │────── requirements.txt            <- package version for installing
    │
    │────── blip2.py                    <- blip2 model
    │────── classifier.py               <- classifier agent
    │────── assistant.py                <- assistant agent
    │────── utils.py                    <- utils
    └────── load.py                     <- file to load embeddings to vector db
--------
