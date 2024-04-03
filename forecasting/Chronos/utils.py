from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from chronos import ChronosPipeline


def preprocess_dataset(dataframe: pd.DataFrame, dynamic_cov: list, time_col: str, target: str) -> pd.DataFrame:
    """
    Receives the raw dataframe and creates:
        - unique id column
        - make dynamic start dates for each series based on the first date where visits is different than 0
        - create 4 static covariates based on the type of tourism
    Args:
        dataframe (pd.DataFrame): raw data
        dynamic_cov (list): column names with dynamic cov
        time_col (str): time column name
        target (str): target name
    Returns:
        pd.DataFrame: cleaned data
    """
    
    # save dynamic cov for later
    dynamic_cov_df = dataframe[dynamic_cov].reset_index().drop_duplicates()
    
    # create target and unique id columns
    dataframe = dataframe.loc[:, ~dataframe.columns.isin(dynamic_cov)].melt(ignore_index=False).reset_index().rename(columns={'variable':'unique_id', 'value':target})
    
    # crete dynamic start dates for each series
    cleaned_df = []
    for i in dataframe['unique_id'].unique():
        temp = dataframe[dataframe['unique_id'] == i]
        cleaned_df.append(temp[temp[time_col] >= min(temp[temp[target] > 0][time_col])])
    cleaned_df = pd.concat(cleaned_df)
    
    # join dynamic cov
    cleaned_df = pd.merge(cleaned_df, dynamic_cov_df, on=[time_col], how='left')
    
    # extract static covariates
    cleaned_df['static_1'] = cleaned_df['unique_id'].apply(lambda x: x[0])
    cleaned_df['static_2'] = cleaned_df['unique_id'].apply(lambda x: x[1])
    cleaned_df['static_3'] = cleaned_df['unique_id'].apply(lambda x: x[2])
    cleaned_df['static_4'] = cleaned_df['unique_id'].apply(lambda x: x[3:])
    
    return cleaned_df


def plot_model_comparison(dataframe: pd.DataFrame) -> None:
    """
    Bar plot comparison between models
    Args:
        dataframe (pd.DataFrame): data with actuals and forecats for both models
    """
    
    dataframe['Date'] = dataframe['Date'].dt.date

    tide_model = dataframe.rename(
        columns={"TiDE": "forecast"}
    )
    tide_model["model"] = "TiDE"
    tide_model["MAPE"] = (
        abs(tide_model["visits"] - tide_model["forecast"]) / tide_model["visits"]
    )

    chronos_tiny_model = dataframe.rename(
        columns={"Chronos Tiny": "forecast"}
    )
    chronos_tiny_model["model"] = "Chronos Tiny"
    chronos_tiny_model["MAPE"] = (
        abs(chronos_tiny_model["visits"] - chronos_tiny_model["forecast"])
        / chronos_tiny_model["visits"]
    )

    chronos_large_model = dataframe.rename(
        columns={"Chronos Large": "forecast"}
    )
    chronos_large_model["model"] = "Chronos Large"
    chronos_large_model["MAPE"] = (
        abs(chronos_large_model["visits"] - chronos_large_model["forecast"])
        / chronos_large_model["visits"]
    )

    plt.rcParams["figure.figsize"] = (20, 5)
    ax = sns.barplot(
        data=pd.concat(
            [tide_model, chronos_tiny_model, chronos_large_model]
        ),
        x="Date",
        y="MAPE",
        hue="model",
        palette=["#dd4fe4", "#070620", "#8a70be", "#fa7302"],
    )
    plt.title("Comparison between TiDE and Chronos in Tourism data")
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.show()


def chronos_forecast(
    model: ChronosPipeline, data: pd.DataFrame, horizon: int, target: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates forecast with Chronos
    Args:
        model (ChronosPipeline): pre-trained model
        data (pd.DataFrame): historical data
        horizon (int): forecast horizon
        target (str): target column
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: lower, mid and upper forecast values
    """
    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    context = torch.tensor(data[target].tolist())
    forecast = model.predict(
        context, horizon
    )  # shape [num_series, num_samples, prediction_length]

    return np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)


def convert_forecast_to_pandas(
    forecast: list, holdout_set: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert forecast to pandas data frame
    Args:
        forecast (list): list with lower, mid and upper bounds
        holdout_set (pd.DataFrame): data frame with dates in forecast horizon
    Returns:
        pd.DataFrame: forecast in pandas format
    """

    forecast_pd = holdout_set[["unique_id", "Date"]]
    forecast_pd.loc[:, "forecast_lower"] = forecast[0]
    forecast_pd.loc[:, "forecast"] = forecast[1]
    forecast_pd.loc[:, "forecast_upper"] = forecast[2]

    return forecast_pd


def plot_actuals_forecast(
    actuals_data: pd.DataFrame, forecast_data: pd.DataFrame, title: str
) -> None:
    """
    Create time series plot actuals vs forecast
    Args:
        actuals_data (pd.DataFrame): actual data
        forecast_data (pd.DataFrame): forecast
        title (str): title for chart
    """

    plt.figure(figsize=(20, 5))
    plt.plot(
        actuals_data["Date"],
        actuals_data["visits"],
        color="royalblue",
        label="historical data",
    )
    plt.plot(
        forecast_data["Date"],
        forecast_data["forecast"],
        color="tomato",
        label="median forecast",
    )
    plt.fill_between(
        forecast_data["Date"],
        forecast_data["forecast_lower"],
        forecast_data["forecast_upper"],
        color="tomato",
        alpha=0.3,
        label="80% prediction interval",
    )
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.show()


def transform_predictions_to_pandas(predictions: list, target: str, pred_list: list, quantiles: list) -> pd.DataFrame:
    """
    Receives as list of predictions and transform it in a data frame
    Args:
        predictions (list): list with predictions
        target (str): column to forecast
        pred_list (list): list with test df to extract time series id
    Returns
        pd.DataFrame: data frame with date, forecast, forecast_lower, forecast_upper and id
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
        temp["unique_id"] = list(pdf.static_covariates_values())[0][0]+list(pdf.static_covariates_values())[0][1]+list(pdf.static_covariates_values())[0][2]+list(pdf.static_covariates_values())[0][3]

        # convert negative predictions into 0
        temp[["forecast", "forecast_lower", "forecast_upper"]] = temp[
            ["forecast", "forecast_lower", "forecast_upper"]
        ].clip(lower=0)

        pred_df_list.append(temp)

    return pd.concat(pred_df_list)
