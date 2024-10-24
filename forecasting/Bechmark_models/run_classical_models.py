import pandas as pd
import os

from load_data.config import DATASETS, DATASETS_FREQ
from load_data.base import *

from statsforecast import StatsForecast
from statsforecast.models import (
    SeasonalNaive,
    AutoETS,
    AutoARIMA,
    RandomWalkWithDrift,
    AutoTheta,
    SimpleExponentialSmoothingOptimized,
    CrostonOptimized
)

for data_name, groups in DATASETS_FREQ.items():
    for group in groups:
        data = DATASETS[data_name]()
        df = data.load_data(group)

        df = df[['unique_id', 'ds', 'y']]

        h_long_term = horizons_long_term[group]
        h_short_term = horizons_short_term[group]

        freq = frequency_pd[group]

        all_results_short_term = []
        all_results_long_term = []

        for name, df_group in df.groupby('unique_id'):
            print(f"{data_name}, {group}, {name}")

            season_len = frequency_map[group]
            if 2 * season_len >= df_group.shape[0]:
                season_len = 2

            models = [
                AutoETS(season_length=season_len),
                AutoARIMA(max_P=1, max_p=1, max_D=1, max_d=1, max_q=1, max_Q=1),
                RandomWalkWithDrift(),
                AutoTheta(season_length=season_len),
                SimpleExponentialSmoothingOptimized(),
                CrostonOptimized(),
                SeasonalNaive(season_length=season_len)
            ]

            sf = StatsForecast(
                models=models,
                freq=group[0],
                n_jobs=-1,
            )

            if df_group.shape[0] > (5 * h_short_term):
                crossvalidation_df_short_term = sf.cross_validation(
                    df=df_group,
                    h=h_short_term,
                    step_size=h_short_term,
                    n_windows=4
                )

                crossvalidation_df_short_term['unique_id'] = name
                all_results_short_term.append(crossvalidation_df_short_term)

            if df_group.shape[0] > (5 * h_long_term):
                crossvalidation_df_long_term = sf.cross_validation(
                    df=df_group,
                    h=h_long_term,
                    step_size=h_long_term,
                    n_windows=4
                )

                crossvalidation_df_long_term['unique_id'] = name
                all_results_long_term.append(crossvalidation_df_long_term)

        if all_results_short_term:
            final_df_short_term = pd.concat(all_results_short_term, ignore_index=True)
            output_filename_short_term = f"results/{data_name}_{group}_short_term_classical.csv"
            os.makedirs(os.path.dirname(output_filename_short_term), exist_ok=True)
            final_df_short_term.to_csv(output_filename_short_term, index=False)

        if all_results_long_term:
            final_df_long_term = pd.concat(all_results_long_term, ignore_index=True)
            output_filename_long_term = f"results/{data_name}_{group}_long_term_classical.csv"
            os.makedirs(os.path.dirname(output_filename_long_term), exist_ok=True)
            final_df_long_term.to_csv(output_filename_long_term, index=False)
