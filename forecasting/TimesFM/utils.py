import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns


def plot_model_comparison(dataframe: pd.DataFrame) -> None:
    """
    Bar plot comparison between models
    Args:
        dataframe (pd.DataFrame): data with actuals and forecats for both models
    """

    dataframe["Date"] = dataframe["Date"].dt.date

    tide_model = dataframe.rename(columns={"TiDE": "forecast"})
    tide_model["model"] = "TiDE"
    tide_model["MAPE"] = (
        abs(tide_model["visits"] - tide_model["forecast"]) / tide_model["visits"]
    )

    moirai_model = dataframe.rename(columns={"MOIRAI": "forecast"})
    moirai_model["model"] = "MOIRAI"
    moirai_model["MAPE"] = (
        abs(moirai_model["visits"] - moirai_model["forecast"]) / moirai_model["visits"]
    )

    chronos_large_model = dataframe.rename(columns={"Chronos Large": "forecast"})
    chronos_large_model["model"] = "Chronos Large"
    chronos_large_model["MAPE"] = (
        abs(chronos_large_model["visits"] - chronos_large_model["forecast"])
        / chronos_large_model["visits"]
    )

    timesfm_model = dataframe.rename(columns={"TimesFM": "forecast"})
    timesfm_model["model"] = "TimesFM"
    timesfm_model["MAPE"] = (
        abs(timesfm_model["visits"] - timesfm_model["forecast"])
        / timesfm_model["visits"]
    )

    plt.rcParams["figure.figsize"] = (20, 5)
    ax = sns.barplot(
        data=pd.concat([tide_model, chronos_large_model, moirai_model, timesfm_model]),
        x="Date",
        y="MAPE",
        hue="model",
        palette=["#dd4fe4", "#070620", "#8a70be", "#fa7302"],
    )
    plt.title("Comparison between TiDE, Chronos, MOIRAI and TimesFM in Tourism data")
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.show()


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
