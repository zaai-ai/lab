from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import MissingValuesFiller
import matplotlib.pyplot as plt
from darts import TimeSeries
import pandas as pd
import numpy as np


def create_dynamic_covariates(train_darts: list, dataframe: pd.DataFrame, forecast_horizon: int,
                              dynamic_covariates_names) -> list:
    dynamic_covariates = []

    # Ensure the Date column is in datetime format in both dataframes
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])

    for serie in train_darts:
        # Extract the unique_id from the series to filter the DataFrame
        unique_id = "".join(str(serie.static_covariates[key].item()) for key in serie.static_covariates)

        # Filter the DataFrame for the current series based on unique_id
        filtered_df = dataframe[dataframe['unique_id'] == unique_id]

        # Generate time-related covariates
        covariate = datetime_attribute_timeseries(
            serie,
            attribute="month",
            one_hot=True,
            cyclic=False,
            add_length=forecast_horizon
        )

        # Add dynamic covariates that need interpolation
        dyn_cov_interp = TimeSeries.from_dataframe(
            filtered_df,
            time_col="Date",
            value_cols=dynamic_covariates_names,
            freq=serie.freq,
            fill_missing_dates=True
        )

        covariate = covariate.stack(MissingValuesFiller().transform(dyn_cov_interp))

        dynamic_covariates.append(covariate)

    return dynamic_covariates


def mape_evaluation(prediction: pd.DataFrame, actuals: pd.DataFrame, target: str) -> list:
    # Convert 'Date' columns to datetime if they aren't already
    prediction['Date'] = pd.to_datetime(prediction['Date'])
    actuals['Date'] = pd.to_datetime(actuals['Date'])

    # Merging prediction and actual sales data on 'Date' and 'unique_id'
    prediction_w_mape = pd.merge(prediction, actuals[['Date', target, 'unique_id']],
                                 on=['Date', 'unique_id'], how='left')

    # Calculating MAPE
    prediction_w_mape['MAPE'] = abs(prediction_w_mape['forecast'] - prediction_w_mape[target]) / \
                                prediction_w_mape[target]

    # Group by 'Date' and calculate the mean MAPE for each group
    weekly_mape = prediction_w_mape.groupby('Date')['MAPE'].mean().tolist()

    # Ensuring the list is rounded to two decimal places
    weekly_mape = [round(x, 2) for x in weekly_mape]

    return weekly_mape


def plot_model_comparison(model_names, model_forecasts, actuals, forecast_horizon, target, top=None):
    if len(model_forecasts) != len(model_names):
        raise ValueError("The number of model forecasts must match the number of model names")

    num_windows = len(model_forecasts[0])
    iteration_mapes = np.zeros((forecast_horizon, len(model_names)))

    # Loop through each model's forecasts
    for model_idx, forecasts in enumerate(model_forecasts):
        for time_window_idx in range(num_windows):
            # Select the correct prediction and actuals based on whether filtering is needed
            model_prediction = forecasts if num_windows == 1 else forecasts[time_window_idx]
            actual_window = actuals if num_windows == 1 else actuals[time_window_idx]

            if top is not None:
                model_prediction = model_prediction[model_prediction['unique_id'].isin(top['unique_id'])]
                actual_window = actual_window[actual_window['unique_id'].isin(top['unique_id'])]

            mape_values = mape_evaluation(model_prediction, actual_window, target)
            iteration_mapes[:, model_idx] += np.array(mape_values)

        iteration_mapes[:, model_idx] /= num_windows

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    indices = np.arange(forecast_horizon)
    bar_width = 0.1

    colors = ['#e854dc', '#ff7404', 'royalblue']

    for i, model in enumerate(model_names):
        ax.bar(indices + i * bar_width, iteration_mapes[:, i], width=bar_width, label=model,color=colors[i])

    ax.set_xlabel('Month')
    ax.set_ylabel('Mean MAPE')
    ax.set_title('Mean MAPE by Model and Month')
    ax.set_yticklabels(['{:.0f}%'.format(x * 100) for x in ax.get_yticks()])
    ax.set_xticks(indices + bar_width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([f'Month {i + 1}' for i in range(forecast_horizon)])
    ax.legend()
    plt.show()


def transform_predictions_to_pandas(predictions, target: str, pred_list: list, quantiles: list,
                                    convert: bool = True) -> pd.DataFrame:
    """
    Receives as list of predictions and transform it in a data frame
    Args:
        predictions (list): list with predictions
        target (str): column to forecast
        pred_list (list): list with test df to extract time series id
    Returns
        pd.DataFrame: data frame with date, forecast, forecast_lower, forecast_upper and id
        :param convert:
    """

    pred_df_list = []

    for p, pdf in zip(predictions, pred_list):
        temp = (
            p.quantile_df(quantiles[1])
            .reset_index()
            .rename(columns={f"{target}_{quantiles[1]}": "forecast"})
        )
        temp["forecast_lower"] = p.quantile_df(quantiles[0]).reset_index()[f"{target}_{quantiles[0]}"]
        temp["forecast_upper"] = p.quantile_df(quantiles[2]).reset_index()[f"{target}_{quantiles[2]}"]

        # add unique id
        unique_id = "".join(str(pdf.static_covariates[key].item()) for key in pdf.static_covariates)

        temp["unique_id"] = unique_id

        if convert:
            # convert negative predictions into 0
            temp[["forecast", "forecast_lower", "forecast_upper"]] = temp[
                ["forecast", "forecast_lower", "forecast_upper"]
            ].clip(lower=0)

        # Reorder columns to make unique_id the first column
        columns_order = ['unique_id'] + [col for col in temp.columns if col != 'unique_id']
        temp = temp[columns_order]

        pred_df_list.append(temp)

    return pd.concat(pred_df_list)
