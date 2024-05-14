# Llama 3

**Steps to run the code:**
1. Create a virtual environment with python 3.10.13
`conda create --name myenv python=3.10.13`
2. Activate in your new virtual environment
`conda activate myenv`
3. Install the required requirements
`pip install -r requirements.txt`
4. Download Llama 3 8B model `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` from https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct-GGUF/tree/main and add it to `llama-3/model/`
5. Download Llama 2 7B model `nous-hermes-llama-2-7b.Q4_K_M.gguf` from https://huggingface.co/TheBloke/Nous-Hermes-Llama-2-7B-GGUF and add it to `llama-3/model/`
6. Run in the notebook `llama.ipynb` 

## Folder Structure:
------------

    ├── llama-3
    │
    ├────────── base                                          <- Configuration class
    ├────────── generator                                     <- Generator class
    ├────────── model                                         <- GGUF models
    │
    │────────── config.yaml                                   <- Config definition
    │
    │────────── llama.ipynb                                   <- notebook
    │────────── utils.py                                      <- helper functions
    │
    └──────── requirements.txt                                <- package versions
--------
