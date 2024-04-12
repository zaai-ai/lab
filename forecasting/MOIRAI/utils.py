from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns


def preprocess_dataset(dataframe: pd.DataFrame, dynamic_cov: list, time_col: str, target: str) -> pd.DataFrame:
    """
    Receives the raw dataframe and creates:
        - unique id column
        - make dynamic start dates for each series based on the first date where visits is different than 0
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

    moirai_model = dataframe.rename(
        columns={"MOIRAI": "forecast"}
    )
    moirai_model["model"] = "MOIRAI"
    moirai_model["MAPE"] = (
        abs(moirai_model["visits"] - moirai_model["forecast"])
        / moirai_model["visits"]
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
            [tide_model, chronos_large_model, moirai_model]
        ),
        x="Date",
        y="MAPE",
        hue="model",
        palette=["#dd4fe4", "#070620", "#8a70be"],
    )
    plt.title("Comparison between TiDE, Chronos and MOIRAI in Tourism data")
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
    
    d = {'unique_id': [],
        time_col: [],
        'forecast': [],
        'forecast_lower': [],
        'forecast_upper': []}
    
    for ts in forecast:
        for j in range(forecast_horizon):
            d['unique_id'].append(ts.item_id)
            d[time_col].append(test_df[test_df['unique_id']==ts.item_id][time_col].tolist()[j])

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


