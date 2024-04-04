# Chronos

**Steps to run the code:**
1. Create a virtual environment with python 3.10.13
`conda create --name myenv python=3.10.13`
3. Activate in your new virtual environment
`conda activate myenv`
4. Install the required requirements:
    - `pip install git+https://github.com/amazon-science/chronos-forecasting.git`
    - `pip install -r requirements.txt`
5. Clone our dataset repository in HF to a different location:
   - `git clone https://huggingface.co/datasets/zaai-ai/time_series_datasets`
   - Move the `data.csv` file to a folder called `data` in this repository
6. Run the notebook

## Folder Structure:
------------

    ├── Chronos
    │
    ├──────── input_data_and_results    <- dataset and results
    │
    │──── requirements.txt              <- package version for installing
    │
    │──── utils.py                      <- helper functions
    └──── Chronos.ipynb                 <- notebook to run the code
--------
