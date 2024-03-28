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

    tide_model = dataframe.rename(
        columns={"TiDE": "forecast"}
    )
    tide_model["model"] = "TiDE"
    tide_model["MAPE"] = (
        abs(tide_model["sales"] - tide_model["forecast"]) / tide_model["sales"]
    )

    chronos_tiny_model = dataframe.rename(
        columns={"Chronos Tiny": "forecast"}
    )
    chronos_tiny_model["model"] = "Chronos Tiny"
    chronos_tiny_model["MAPE"] = (
        abs(chronos_tiny_model["sales"] - chronos_tiny_model["forecast"])
        / chronos_tiny_model["sales"]
    )

    chronos_large_model = dataframe.rename(
        columns={"Chronos Large": "forecast"}
    )
    chronos_large_model["model"] = "Chronos Large"
    chronos_large_model["MAPE"] = (
        abs(chronos_large_model["sales"] - chronos_large_model["forecast"])
        / chronos_large_model["sales"]
    )

    plt.rcParams["figure.figsize"] = (20, 5)
    ax = sns.barplot(
        data=pd.concat(
            [tide_model, chronos_tiny_model, chronos_large_model]
        ),
        x="year_month",
        y="MAPE",
        hue="model",
        palette=["#dd4fe4", "#070620", "#8a70be"],
    )
    plt.title("Comparison between TiDE and Chronos in EV data")
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
    context = torch.tensor(data["sales"].tolist())
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

    forecast_pd = holdout_set[["Car_category", "year_month"]]
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
        actuals_data["year_month"],
        actuals_data["sales"],
        color="royalblue",
        label="historical data",
    )
    plt.plot(
        forecast_data["year_month"],
        forecast_data["forecast"],
        color="tomato",
        label="median forecast",
    )
    plt.fill_between(
        forecast_data["year_month"],
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
        temp["Car_category"] = list(pdf.static_covariates_values())[0][0]

        # convert negative predictions into 0
        temp[["forecast", "forecast_lower", "forecast_upper"]] = temp[
            ["forecast", "forecast_lower", "forecast_upper"]
        ].clip(lower=0)

        pred_df_list.append(temp)

    return pd.concat(pred_df_list)
