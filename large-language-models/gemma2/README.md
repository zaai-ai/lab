# Gemma

**Steps to run the code:**
1. Create a virtual environment with python 3.10.13
`conda create --name myenv python=3.10.13`
2. Activate in your new virtual environment
`conda activate myenv`
3. Install the required requirements
`pip install -r requirements.txt`
4. Download Gemma 7B model `gemma-7b-it-Q4_K_M.gguf` from https://huggingface.co/rahuldshetty/gemma-7b-it-gguf-quantized/tree/main and add it to `gemma2/model/`
5. Download Gemma 2 9B model `gemma-2-9b-it-Q4_K_M` from https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/tree/main and add it to `gemma2/model/`
6. Create a folder called /env under gemma2/ and add a file with the following:
    - var.env
    ```
    OPENAI_KEY=YOUR_OPENAI_KEY
    ```
7. Run in the notebook `gemma2.ipynb` 

## Folder Structure:
------------

    ├── gemma2
    │
    ├────────── base                                          <- Configuration class
    ├────────── generator                                     <- Generator class
    ├────────── model                                         <- GGUF models
    │
    │────────── config.yaml                                   <- Config definition
    │
    │────────── gemma2.ipynb                                  <- notebook
    │────────── utils                                         <- helper functions
    │
    └──────── requirements.txt                                <- package versions
--------
