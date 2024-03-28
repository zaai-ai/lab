from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns


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

    chronos_large_model = dataframe.rename(
        columns={"Chronos Large": "forecast"}
    )
    chronos_large_model["model"] = "Chronos Large"
    chronos_large_model["MAPE"] = (
        abs(chronos_large_model["sales"] - chronos_large_model["forecast"])
        / chronos_large_model["sales"]
    )
    
    moirai_model = dataframe.rename(
        columns={"MOIRAI": "forecast"}
    )
    moirai_model["model"] = "MOIRAI"
    moirai_model["MAPE"] = (
        abs(moirai_model["sales"] - moirai_model["forecast"])
        / moirai_model["sales"]
    )

    plt.rcParams["figure.figsize"] = (20, 5)
    ax = sns.barplot(
        data=pd.concat(
            [tide_model, chronos_large_model, moirai_model]
        ),
        x="year_month",
        y="MAPE",
        hue="model",
        palette=["#dd4fe4", "#070620", "#fa7302"],
    )
    plt.title("Comparison between TiDE, Chronos and MOIRAI in EV data")
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.show()


def moirai_forecast_to_pandas(forecast, test_df: pd.DataFrame, forecast_horizon: int, time_col: str) -> pd.DataFrame:
    """
    Converts MOIRAI forecast into pandas dataframe
    Args:
        forecast: MOIRAI's forecast
        test_df: dataframe with actuals
        forecast_horizon: forecast horizon
        time_col: date column
    Returns:
        pd.DataFrame: forecast in pandas format
    """
    
    d = {'Car_category': [],
        time_col: [],
        'forecast': [],
        'forecast_lower': [],
        'forecast_upper': []}
    
    for ts in forecast:
        for j in range(forecast_horizon):
            d['Car_category'].append(ts.item_id)
            d[time_col].append(test_df[test_df['Car_category']==ts.item_id][time_col].tolist()[j])

            temp = []
            for i in range(ts.samples.shape[0]):
                temp.append(ts.samples[i][j])

            d["forecast"].append(np.median(temp))
            d["forecast_lower"].append(
                np.percentile(temp, 10)
            )
            d["forecast_upper"].append(
                np.percentile(temp, 90)
            )
    
    return pd.DataFrame(d)




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


