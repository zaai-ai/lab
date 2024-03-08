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

    plt.rcParams["figure.figsize"] = (20, 5)
    ax = sns.barplot(
        data=pd.concat([tide_model, timegpt_model]),
        x="delivery_week",
        y="MAPE",
        hue="model",
        palette=["#dd4fe4", "#070620"],
    )
    plt.title("Comparison between TiDE and TimeGPT models in real data")
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.show()
