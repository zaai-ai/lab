from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.pipeline import Pipeline
from dateutil.relativedelta import relativedelta
from darts.models import TiDEModel
from multiprocessing import Pool
from darts.metrics import rmse
from darts import TimeSeries
from sys import argv
import pandas as pd
import numpy as np
import warnings
import random
import torch
import utils

torch.set_float32_matmul_precision('high')
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")


DATASET = "hf://datasets/zaai-ai/time_series_datasets/data.csv"

NUM_FOLDS = 4

TIME_COL = "Date"
TARGET = "visits"
RESIDUALS_TARGET = "residuals"
STATIC_COV = ["static_1", "static_2", "static_3", "static_4"]
DYNAMIC_COV = ["CPI", "Inflation_Rate", "GDP"]

FORECAST_HORIZON = 6  # months
FREQ = "MS"

SCALER = Scaler()
TRANSFORMER = StaticCovariatesTransformer()
PIPELINE = Pipeline([SCALER, TRANSFORMER])


def save_dict_to_txt(rmse, data_dict, file_name):
    with open(file_name, 'w') as f:
        f.write(f'RMSE: {rmse}\n\n\n\n')
        for key, v in data_dict.items():
            # Convert value to a string if it's not already
            value_str = str(v)
            f.write(f'{key}: {value_str}\n')

    print(f"\nBest params has been saved to '{file_name}'.\n")


def train_model(params, fold):
    residuals = pd.read_csv('data/residuals.csv')
    residuals[TIME_COL] = pd.to_datetime(residuals[TIME_COL])

    residuals_train = residuals[residuals[TIME_COL] <= fold['start_date']]
    residuals_df = residuals[(residuals[TIME_COL] <= fold['end_date'])]

    residuals_darts = TimeSeries.from_group_dataframe(
        df=residuals_train,
        group_cols=STATIC_COV,
        time_col=TIME_COL,
        value_cols=RESIDUALS_TARGET,
        freq=FREQ,
        fill_missing_dates=True,
        fillna_value=0)

    dynamic_covariates = utils.create_dynamic_covariates(residuals_darts, residuals_df, FORECAST_HORIZON, DYNAMIC_COV)

    # scale covariates
    dynamic_covariates_transformed = SCALER.fit_transform(dynamic_covariates)

    # scale data and transform static covariates
    data_transformed = PIPELINE.fit_transform(residuals_darts)

    residuals_tide = TiDEModel(**params)
    residuals_tide.fit(data_transformed, future_covariates=dynamic_covariates_transformed, verbose=False)
    pred = PIPELINE.inverse_transform(residuals_tide.predict(n=FORECAST_HORIZON, series=data_transformed,
                                                             future_covariates=dynamic_covariates_transformed,
                                                             num_samples=50))
    residuals_forecast = utils.transform_predictions_to_pandas(pred, RESIDUALS_TARGET, residuals_darts,
                                                               [0.25, 0.5, 0.75], convert=False)
    prediction = utils.combine_predictions(fold['predictions']['chronos'], residuals_forecast)

    # Prepare lists for TimeSeries objects
    forecast_ts_list = []
    test_ts_list = []

    # Obtain unique IDs and ensure consistent ordering
    top = df.groupby(['unique_id']).agg({TARGET: 'sum'}).reset_index().sort_values(by=TARGET, ascending=False).head(25)
    unique_ids = sorted(top['unique_id'].unique())

    for uid in unique_ids:
        # Prepare forecast TimeSeries
        forecast_df = prediction[prediction['unique_id'] == uid]
        forecast_ts = TimeSeries.from_dataframe(forecast_df, time_col=TIME_COL, value_cols=['forecast'], freq=FREQ)
        forecast_ts_list.append(forecast_ts)

        # Prepare corresponding test TimeSeries
        test_df = fold['test'][fold['test']['unique_id'] == uid]
        test_ts = TimeSeries.from_dataframe(test_df, time_col=TIME_COL, value_cols=TARGET, freq=FREQ)
        test_ts_list.append(test_ts)

    return forecast_ts_list, test_ts_list


def calculate_rmse_for_fold(args):
    params, fold = args
    rmse_value_mean = 0
    forecast_ts_list, test_ts_list = train_model(params, fold)
    for forecast_ts, test_ts in zip(forecast_ts_list, test_ts_list):
        rmse_value_mean += rmse(test_ts, forecast_ts, series_reduction=np.nanmean)
    rmse_value_mean /= len(forecast_ts_list)
    return rmse_value_mean


def find_best_params(folds):
    lowest_rmse = float('inf')
    start_timestamp = pd.Timestamp.now()

    print("\n\nStarting hyperparameter tuning: \n\n")

    while True:
        params = get_tide_params()
        rmse_value = 0

        with Pool(NUM_FOLDS) as p:
            rmse_values = p.map(calculate_rmse_for_fold, [(params, fold) for fold in folds])
            rmse_value = np.mean(rmse_values)

        if rmse_value < lowest_rmse:
            lowest_rmse = rmse_value
            save_dict_to_txt(rmse_value, params, 'data/best_params.txt')

        print(f"\n\n\nRMSE: {rmse_value} - Best RMSE: {lowest_rmse}")
        print(f"Time elapsed: {pd.Timestamp.now() - start_timestamp}\n\n\n")


def get_tide_params():
    return {
        "input_chunk_length": random.choice([2, 3, 4, 6, 7]),
        "output_chunk_length": FORECAST_HORIZON,
        "num_encoder_layers": random.choice([2, 4, 6, 8]),
        "num_decoder_layers": random.choice([2, 4, 6, 8]),
        "decoder_output_dim": random.choice([6, 8, 10, 15, 16]),
        "hidden_size": random.choice([2, 4, 8, 16]),
        "temporal_width_past": random.choice([2, 4, 8]),
        "temporal_width_future": random.choice([4, 8, 10, 12]),
        "temporal_decoder_hidden": random.choice([16, 23, 26, 32]),
        "dropout": random.choice([0.1, 0.15, 0.3]),
        "batch_size": random.choice([8, 16, 32, 64]),
        "n_epochs": random.choice([10, 15, 20, 25, 30]),
        "likelihood": QuantileRegression(quantiles=[0.25, 0.5, 0.75]),
        "random_state": 42,
        "use_static_covariates": True,
        "optimizer_kwargs": {"lr": random.choice([1e-3, 1e-4, 1e-5, 1e-6])},
        "use_reversible_instance_norm": random.choice([True, False]),
    }


def test_tide_model():
    # RMSE: 138.12623755006888
    return {
        'input_chunk_length': 7,
        'output_chunk_length': FORECAST_HORIZON,
        'num_encoder_layers': 4,
        'num_decoder_layers': 6,
        'decoder_output_dim': 15,
        'hidden_size': 4,
        'temporal_width_past': 4,
        'temporal_width_future': 10,
        'temporal_decoder_hidden': 23,
        'dropout': 0.1,
        'batch_size': 64,
        'n_epochs': 30,
        'likelihood': QuantileRegression(quantiles=[0.25, 0.5, 0.75]),
        'random_state': 42,
        'use_static_covariates': True,
        'optimizer_kwargs': {'lr': 0.001},
        'use_reversible_instance_norm': True,
    }


if __name__ == '__main__':
    # load data
    df = pd.read_csv(DATASET).drop(columns=["Unnamed: 0"])
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[TARGET] = df[TARGET].replace(0.0, 0.01)
    df[TARGET] = df[TARGET].round(2)

    end_date = df[TIME_COL].max()

    folds = []
    j = 1
    for i in range(NUM_FOLDS, 0, -1):
        start_date = end_date - relativedelta(months=FORECAST_HORIZON * i)
        end_date_fold = start_date + relativedelta(months=FORECAST_HORIZON)

        fold = {
            "start_date": start_date,
            "end_date": end_date_fold,
            "train": None,
            "test": None,
            "df": None,
            "predictions": {
                "chronos": pd.read_csv(f'data/chronos_forecast_fold{j}.csv'),
                "tide": None,
                "hybrid": {
                    "default": None,
                    "tuned": None
                }
            }
        }
        fold["predictions"]["chronos"][TIME_COL] = pd.to_datetime(fold["predictions"]["chronos"][TIME_COL])
        j += 1

        fold["train"] = df[df[TIME_COL] <= fold["start_date"]]
        fold["test"] = df[(df[TIME_COL] > fold["start_date"]) & (df[TIME_COL] <= fold["end_date"])]
        fold["df"] = df[df[TIME_COL] <= fold["end_date"]]
        folds.append(fold)

    if len(argv) > 1:
        if argv[1] == 'test':
            print(test_tide_model())
        exit(0)

    find_best_params(folds)
