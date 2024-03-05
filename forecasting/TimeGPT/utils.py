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

    zaai_model = dataframe.drop(columns="TimeGPT")
    zaai_model["model"] = "ZAAI"
    zaai_model["MAPE"] = (
        abs(zaai_model["target"] - zaai_model["forecast"]) / zaai_model["target"]
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
        data=pd.concat([zaai_model, timegpt_model]),
        x="delivery_week",
        y="MAPE",
        hue="model",
        palette=["#dd4fe4", "#070620"],
    )
    plt.title("Comparison between ZAAI and TimeGPT models in real data")
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.show()
