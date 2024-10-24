import pandas as pd
import os
import torch
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import timesfm
from load_data.base import *
from load_data.config import DATASETS,DATASETS_FREQ

n_windows = 4
UNIQUE_ID_COL = 'item_id'
TIME_COL = 'timestamp'

for data_name, groups in DATASETS_FREQ.items():
    for group in groups:
        data = DATASETS[data_name]()
        df = data.load_data(group)

        df = df[['unique_id', 'ds', 'y']]
        df['ds'] = pd.to_datetime(df['ds'])

        df_ts = TimeSeriesDataFrame(df, id_column='unique_id', timestamp_column='ds')
        df_ts = df_ts.rename(columns={'y': 'target'})

        h_long_term = horizons_long_term[group]
        h_short_term = horizons_short_term[group]

        for horizon_type, horizon in [('short_term', h_short_term), ('long_term', h_long_term)]:
            all_results = []

            for window in range(1, n_windows + 1):
                test = df.groupby('unique_id').tail(horizon * window)
                train = df.drop(test.index)
                test = test.groupby('unique_id').head(horizon)

                test_ts = df_ts.groupby(UNIQUE_ID_COL).tail(horizon * window)
                train_ts = df_ts.drop(test_ts.index)
                test_ts = test_ts.groupby(UNIQUE_ID_COL).head(horizon)

                cutoff_date = train['ds'].iloc[-1]
                test = test.copy()
                test.loc[:, 'cutoff'] = cutoff_date

                chronos_predictions_list = []

                # ------------------------- Chronos Model -------------------------
                predictor = TimeSeriesPredictor(prediction_length=horizon).fit(train_ts, presets="chronos_tiny",)
                predictions = predictor.predict(train_ts)
                predictions_df = pd.DataFrame(predictions)
                predictions_df = predictions_df.reset_index()
                predictions_df = predictions_df.rename(columns={'item_id': 'unique_id', 'timestamp': 'ds'})
                test = test.merge(predictions_df[['unique_id', 'ds', 'mean']], on=['unique_id', 'ds'], how='left')

                # ------------------------- TimesFM Model -------------------------
                tfm = timesfm.TimesFm(
                    context_len=512,
                    horizon_len=horizon,
                    input_patch_len=32,
                    output_patch_len=128,
                    num_layers=20,
                    model_dims=1280,
                    backend='cpu',
                )
                tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

                forecast_df = tfm.forecast_on_df(
                    inputs=train,
                    freq=group[0],
                    value_name="y",
                    num_jobs=-1,
                )

                test = test.merge(forecast_df[['unique_id', 'ds', 'timesfm']], on=['unique_id', 'ds'], how='left')
                test.to_csv(f"results/{data_name}_{group}_{horizon_type}_{window}_foundation.csv")
                all_results.append(test)

            # ------------------------- Combine Results -------------------------
            final_combined_results = pd.concat(all_results, ignore_index=True)
            final_combined_results = final_combined_results.reset_index()

            output_filename = f"results/{data_name}_{group}_{horizon_type}_foundation.csv"
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            final_combined_results.to_csv(output_filename, index=False)
