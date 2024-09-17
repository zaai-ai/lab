# RAQ

**Steps to run the code:**
1. Create a virtual environment with python 3.10.13
`conda create --name myenv python=3.10.13`
2. Activate in your new virtual environment
`conda activate myenv`
3. Install the required requirements
`pip install -r requirements.txt`
4. Create a folder called `/env` under `RAQ/` and add a file with the following:
    - var.env
    ```
    OPENAI_KEY=YOUR_OPENAI_KEY
    NGC_API_KEY=YOUR_NGC_API_KEY
    ANTHROPIC_KEY=YOUR_ANTHROPIC_KEY
    GEMINI_KEY=YOUR_GEMINI_KEY
    ```
5. Run in the notebook `raq.ipynb` 

## Folder Structure:
------------

    ├── RAQ
    │
    ├────────── env                                           <- env files
    │
    │────────── raq.ipynb                                     <- notebook
    │────────── utils.py                                      <- helper functions
    │────────── generator.py                                  <- class to parse questions and get answer from LLM
    │
    └──────── requirements.txt                                <- package versions
--------
