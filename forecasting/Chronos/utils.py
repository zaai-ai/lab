from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from chronos import ChronosPipeline


def plot_model_comparison(dataframe: pd.DataFrame) -> None:
    """
    Bar plot comparison between models
    Args:
        dataframe (pd.DataFrame): data with actuals and forecats for both models
    """

    tide_model = dataframe.drop(columns="TimeGPT")
    tide_model["model"] = "TiDE"
    tide_model["MAPE"] = (
        abs(tide_model["target"] - tide_model["forecast"]) / tide_model["target"]
    )

    timegpt_model = dataframe.drop(columns="forecast").rename(
        columns={"TimeGPT": "forecast"}
    )
    timegpt_model["model"] = "TimeGPT"
    timegpt_model["MAPE"] = (
        abs(timegpt_model["target"] - timegpt_model["forecast"])
        / timegpt_model["target"]
    )

    chronos_tiny_model = dataframe.drop(columns="forecast").rename(
        columns={"Chronos Tiny": "forecast"}
    )
    chronos_tiny_model["model"] = "Chronos Tiny"
    chronos_tiny_model["MAPE"] = (
        abs(chronos_tiny_model["target"] - chronos_tiny_model["forecast"])
        / chronos_tiny_model["target"]
    )

    chronos_large_model = dataframe.drop(columns="forecast").rename(
        columns={"Chronos Large": "forecast"}
    )
    chronos_large_model["model"] = "Chronos Large"
    chronos_large_model["MAPE"] = (
        abs(chronos_large_model["target"] - chronos_large_model["forecast"])
        / chronos_large_model["target"]
    )

    plt.rcParams["figure.figsize"] = (20, 5)
    ax = sns.barplot(
        data=pd.concat(
            [tide_model, timegpt_model, chronos_tiny_model, chronos_large_model]
        ),
        x="delivery_week",
        y="MAPE",
        hue="model",
        palette=["#dd4fe4", "#070620", "#8a70be", "#fa7302"],
    )
    plt.title("Comparison between TiDE, TimeGPT and Chronos in real data")
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.show()


def chronos_forecast(
    model: ChronosPipeline, data: pd.DataFrame, horizon: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates forecast with Chronos
    Args:
        model (ChronosPipeline): pre-trained model
        data (pd.DataFrame): historical data
        horizon (int): forecast horizon
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: lower, mid and upper forecast values
    """
    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    context = torch.tensor(data["target"].tolist())
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

    forecast_pd = holdout_set[["unique_id", "delivery_week"]]
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
        actuals_data["delivery_week"],
        actuals_data["target"],
        color="royalblue",
        label="historical data",
    )
    plt.plot(
        forecast_data["delivery_week"],
        forecast_data["forecast"],
        color="tomato",
        label="median forecast",
    )
    plt.fill_between(
        forecast_data["delivery_week"],
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
