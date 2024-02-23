# FinalMLP

**Steps to run the code:**
1. Create a virtual environment with python 3.7.16
`conda create --name myenv python=3.7.16`
2. Activate in your new virtual environment
`conda activate myenv`
3. Install the required requirements
`pip install -r requirements.txt`
4. Create a folder called `/data/<YOUR USE CASE>` under `FinalMLP/` and add your data
5. Create a folder called `/config/<YOUR USE CASE>` and add 2 files with the following:
    - dataset_config.yaml
    - model_config.yaml
6. Run the notebook

## Folder Structure:
------------

    ├── FinalMLP
    │
    ├──────── checkpoints/<YOUR USE CASE>                   <- Automatically created when training the model
    ├──────── config/<YOUR USE CASE>                        <- yaml files with dataset and model config
    ├──────── data                                          <- folder with data and feature definitions for several use cases
    ├───────────── <YOUR USE CASE>
    ├───────────────── <YOUR USE CASE>                      <- feature definitions and h5 files
    ├───────────────── csv files                            <- train, valid and test csv files
    ├──────── src                                           <- FinalMLP and DualMLP code
    │
    │──── requirements.txt                                  <- package version for installing
    │
    └──── FinalMLP.ipynb                                    <- notebook to run the code
--------


## Configuration Guide

  
The `dataset_config.yaml` file contains all the dataset settings as follows.
  
| Params                        | Type | Default | Description                                                                                                                             |
| ----------------------------- | ---- | ------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| data_root                     | str  |         | the root directory to load and save data data                                                                                                          |
| data_format                   | str  |         | input data format, "h5", "csv", or "tfrecord" supported                                                                                 |
| train_data                    | str  | None    | training data path                                                                                                                      |
| valid_data                    | str  | None    | validation data path                                                                                                                    |
| test_data                     | str  | None    | test data path                                                                                                                          |
| min_categr_count              | int  | 1       | min count to filter category features,                                                                                                  |
| feature_cols                  | list |         | a list of features with the following dict keys                                                                                         |
| feature_cols::name            | str\|list  |         | feature column name in csv. A list is allowed in which the features have the same feature type and will be expanded accordingly.                                                                                                               |
| feature_cols::active          | bool |         | whether to use the feature                                                                                                              |
| feature_cols::dtype           | str  |         | the input data dtype, "int"\|"str"                                                                                                       |
| feature_cols::type            | str  |         | feature type "numeric"\|"categorical"\|"sequence"\|"meta"                                                                                  |
| label_col                     | dict |         | specify label column                                                                                                                    |
| label_col::name               | str  |         | label column name in csv                                                                                                                |
| label_col::dtype              | str  |         | label data dtype                                                                                                                        |



The `model_config.yaml` file contains all the model hyper-parameters as follows.
  
| Params                  | Type            | Default                 | Description                                                                                                                                                                                                       |
| ----------------------- | --------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model                   | str             | "FinalMLP"              | model name,  which should be same with model class name                                                                                                                                                           |
| dataset_id              | str             | "TBD"                   | dataset_id to be determined                                                                                                                                                                                       |
| loss                    | str             | "binary_crossentropy"   | loss function                                                                                                                                                                                                     |
| metrics                 | list            | ['logloss', 'AUC']      | a list of metrics for evaluation                                                                                                                                                                                  |
| task                    | str             | "binary_classification" | task type supported: ```"regression"```, ```"binary_classification"```                                                                                                                                            |
| optimizer               | str             | "adam"                  | optimizer used for training                                                                                                                                                                                       |
| learning_rate           | float           | 1.0e-3                  | learning rate                                                                                                                                                                                                     |
| embedding_regularizer   | float\|str       | 0                       | regularization weight for embedding matrix: L2 regularization is applied by default. Other optional examples: ```"l2(1.e-3)"```, ```"l1(1.e-3)"```, ```"l1_l2(1.e-3, 1.e-3)"```.                                  |
| net_regularizer         | float\|str       | 0                       | regularization weight for network parameters: L2 regularization is applied by default. Other optional examples: ```"l2(1.e-3)"```, ```"l1(1.e-3)"```, ```"l1_l2(1.e-3, 1.e-3)"```.                                |
| batch_size              | int             | 10000                   | batch size, usually a large number for CTR prediction task                                                                                                                                                        |
| embedding_dim           | int             | 32                      | embedding dimension of features. Note that field-wise embedding_dim can be specified in ```feature_specs```.                                                                                                      |
| mlp1_hidden_units       | list            | [64, 64, 64]            | hidden units in MLP1                                                                                                                                                                                              |
| mlp1_hidden_activations | str\|list        | "relu"                  | activation function in MLP1. Particularly, layer-wise activations can be specified as a list, e.g., ["relu",  "leakyrelu", "sigmoid"]                                                                             |
| mlp2_hidden_units       | list            | [64, 64, 64]            | hidden units in MLP2                                                                                                                                                                                              |
| mlp2_hidden_activations | str             | "relu"                  | activation function in MLP2. Particularly, layer-wise activations can be specified as a list, e.g., ["relu", "leakyrelu", "sigmoid"]                                                                              |
| mlp1_dropout            | float           | 0                       | dropout rate in MLP1                                                                                                                                                                                              |
| mlp2_dropout            | float           | 0                       | dropout rate in MLP2                                                                                                                                                                                              |
| mlp1_batch_norm         | bool            | False                   | whether using BN in MLP1                                                                                                                                                                                          |
| mlp2_batch_norm         | bool            | False                   | whether using BN in MLP2                                                                                                                                                                                          |
| use_fs                  | bool            | True                    | whether using feature selection                                                                                                                                                                                   |
| fs_hidden_units         | list            | [64]                    | hidden units of fs gates                                                                                                                                                                                          |
| fs1_context             | list            | []                      | conditional features for feature gating in stream 1                                                                                                                                                               |
| fs2_context             | list            | []                      | conditional features for feature gating in stream 2                                                                                                                                                               |
| num_heads               | int             | 1                       | number of heads used for bilinear fusion                                                                                                                                                                          |
| epochs                  | int             | 100                     | the max number of epochs for training, which can early stop via monitor metrics.                                                                                                                                  |
| shuffle                 | bool            | True                    | whether shuffle the data samples for each epoch of training                                                                                                                                                       |
| seed                    | int             | 2021                    | the random seed used for reproducibility                                                                                                                                                                          |
| monitor                 | str\|dict        | 'AUC'                   | the monitor metrics for early stopping. It supports a single metric, e.g., ```"AUC"```. It also supports multiple metrics using a dict, e.g., {"AUC": 2, "logloss": -1} means ```2*AUC - logloss```.              |
| monitor_mode            | str             | 'max'                   | ```"max"``` means that the higher the better, while ```"min"``` denotes that the lower the better.                                                                                                                |
| model_root              | str             | './checkpoints/'        | the dir to save model checkpoints and running logs                                                                                                                                                                |
| early_stop_patience     | int             | 2                       | training is stopped when monitor metric fails to become better for ```early_stop_patience=2```consective evaluation intervals.                                                                                    |
| save_best_only          | bool            | True                    | whether to save the best model checkpoint only                                                                                                                                                                    |
| eval_steps              | int\|None        | None                    | evaluate the model on validation data every ```eval_steps```. By default, ```None``` means evaluation every epoch.                                                                                                |


