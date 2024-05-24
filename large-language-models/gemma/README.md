# Gemma

**Steps to run the code:**
1. Create a virtual environment with python 3.10.13
`conda create --name myenv python=3.10.13`
2. Activate in your new virtual environment
`conda activate myenv`
3. Install the required requirements
`pip install -r requirements.txt`
4. Download Mistral 7B model `mistral-7b-instruct-v0.1.Q4_K_M.gguf` from https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main and add it to `gemma/model/`
5. Download Llama 3 8B model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf from https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct-GGUF/tree/main and add it to `gemma/model/`
6. Download Gemma 7B model `gemma-7b-it-Q4_K_M.gguf` from https://huggingface.co/rahuldshetty/gemma-7b-it-gguf-quantized/tree/main and add it to `gemma/model/`
7. Run in the notebook `gemma.ipynb` 

## Folder Structure:
------------

    ├── gemma
    │
    ├────────── base                                          <- Configuration class
    ├────────── generator                                     <- Generator class
    ├────────── model                                         <- GGUF models
    │
    │────────── config.yaml                                   <- Config definition
    │
    │────────── gemma.ipynb                                   <- notebook
    │────────── utils.py                                      <- helper functions
    │
    └──────── requirements.txt                                <- package versions
--------
