# NVIDIA NIM & RAQ

**Steps to run the code:**
1. Create a virtual environment with python 3.10.13
`conda create --name myenv python=3.10.13`
2. Activate in your new virtual environment
`conda activate myenv`
3. Install the required requirements
`pip install -r requirements.txt`
4. Create a folder called `/env` under `nim-nvidia-raq/` and add a file with the following:
    - var.env
    ```
    OPENAI_KEY=YOUR_OPENAI_KEY
    NGC_API_KEY=YOUR_NGC_API_KEY
    ```
5. Run in the notebook `nim-raq.ipynb` 

## Folder Structure:
------------

    ├── nim-nvidia-raq
    │
    ├────────── env                                           <- env files
    │
    │────────── nim-raq.ipynb                                 <- notebook
    │────────── utils.py                                      <- helper functions
    │────────── generator.py                                  <- class to parse questions and get answer from LLM
    │
    └──────── requirements.txt                                <- package versions
--------
