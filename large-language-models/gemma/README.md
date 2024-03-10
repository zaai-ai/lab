# Gemma

**Steps to run the code:**
1. Create a virtual environment with python 3.10.13
`conda create --name myenv python=3.10.13`
3. Activate in your new virtual environment
`conda activate myenv`
4. Install the required requirements
`pip install -r requirements.txt`
5. Create a folder called `/env` under `gemma/` and add a file with the following:
    - openai.env
    ```
    OPENAI_KEY=YOUR_OPENAI_KEY
    ```
6. Download Mistral 7B model `mistral-7b-instruct-v0.1.Q4_K_M.gguf` from https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main and add it to `gemma/model/`
7. Download Llama 7B model `nous-hermes-llama-2-7b.Q4_K_M.gguf` from https://huggingface.co/TheBloke/Nous-Hermes-Llama-2-7B-GGUF and add it to `gemma/model/`
8. Download Gemma 7B model `gemma-7b-it-Q4_K_M.gguf` from https://huggingface.co/rahuldshetty/gemma-7b-it-gguf-quantized/tree/main and add it to `gemma/model/`
9. Run in the notebook `gemma.ipynb` 

## Folder Structure:
------------

    ├── gemma
    │
    ├────────── base                                          <- Configuration class
    ├────────── generator                                     <- Generator class
    ├────────── env                                           <- env files
    ├────────── model                                         <- GGUF models
    │
    │────────── config.yaml                                   <- Config definition
    │
    │────────── gemma.ipynb                                   <- notebook
    │────────── utils.py                                      <- helper functions
    │
    └──────── requirements.txt                                <- package versions
--------
