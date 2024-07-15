# ZAAI
Curricular Internship at ZAAI

# Summary
The goal of this project is to compared the models to a hybrid combination of both of them to see if an improved prediction can be made.
The models used are:

- Chronos
- TiDE

Initially, we'll generate predictions using Chronos and then employ its residuals as training input for the TiDE model. Then use these residuals predictions to refine the original Chronos forecast.

In the end we'll compare the models using the Mean Absolute Percentage Error (MAPE) metric. 

# Versions

The version of the operating system used to develop this project is:
- macOS Sonoma 14.5
- Windows 11

Python Versions:
- 3.12

# Requirements

To keep everything organized and simple,
we will use [MiniConda](https://docs.conda.io/projects/miniconda/en/latest/) to manage our environments.

To create an environment with the required packages for this project, run the following commands:

```bash
conda create -n venv python
```

Then we need to install the requirements:

```bash
pip install -r requirements.txt
```

if there is a bug with lightbm do this:
- `brew install libomp`

# Results

You can see the notebook here: [notebook.ipynb](internship.ipynb).


