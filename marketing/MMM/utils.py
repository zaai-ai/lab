import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM
from typing import Tuple


def line_plot(dataframe: pd.DataFrame, features: list, title: str) -> None:
    """
    Plots multiple time series
    Args:
        dataframe (pd.DataFrame): data
        features (list): features to plot
        title (str): title for chart
    """

    colors = {0: '#070620', 1: '#dd4fe4'}
    
    plt.rcParams["figure.figsize"] = (20,3)
    for i in range(len(features)):
        dataframe[[features[i]]] = MinMaxScaler().fit_transform(dataframe[[features[i]]])
        sns.lineplot(data=dataframe, x='ds', y=features[i], label=features[i], color=colors[i])
    plt.legend()
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Normalized Values')
    plt.show()


def get_sigma_for_beta_channels(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Estimates the standard deviation value for the prior distribution of beta based on the spends per channel
    Args:
        dataframe (pd.DataFrame): spends per channel
    Returns:
        np.ndarray: mu for each channel
    """

    total_spend_per_channel = dataframe.sum(axis=0)
    spend_share = total_spend_per_channel / total_spend_per_channel.sum()
    # The scale necessary to make a HalfNormal distribution have unit variance
    HALFNORMAL_SCALE = 1 / np.sqrt(1 - 2 / np.pi)
    n_channels = 3
    
    return HALFNORMAL_SCALE * n_channels * spend_share.to_numpy()


def extract_trend_seasonality(dataframe: pd.DataFrame, target: str, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract seasonality and trend with Prophet
    Args:
        dataframe (pd.DataFrame): data
        target (str): target variable
        horizon (int): forecast horizon
    Returns:
        Tuple[np.ndarray, np.ndarray]: seasonality and trend
    """

    m = Prophet()
    m.fit(dataframe[['ds', target]].rename(columns={target:'y'}))
    future = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)
    m.plot_components(forecast)

    return forecast['yearly'].values.ravel(), forecast['trend'].values.ravel()


def plot_ROAS(model: DelayedSaturatedMMM, dataframe: pd.DataFrame, channels: list) -> None:
    """
    Plot return od ad spend
    Args:
        model (DelayedSaturatedMMM): fitted mmm model 
        dataframe (pd.DataFrame): training data
        channels (list): media channels
    """
    channel_contribution_original_scale = model.compute_channel_contribution_original_scale()
    roas_samples = (
        channel_contribution_original_scale.stack(sample=("chain", "draw")).sum("date")
        / dataframe[channels].sum().to_numpy()[..., None]
    )

    roas_dict = {}
    for i in channels:
        roas_dict[i] = roas_samples.sel(channel=i).to_numpy()
    roas_df = pd.DataFrame(roas_dict)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in channels:
        sns.histplot(data=roas_df, x=i, alpha=0.3, kde=True, label=i, ax=ax)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set(title="Posterior ROAS distribution", xlabel="ROAS")
    plt.show()


def define_pipeline(
        numerical_cols: list,
    ) -> Pipeline:
        """
        Defines data transformation pipeline
        Args:
            numerical_cols (list): list of numerical features to be scaled
        Returns:
            pipe (sk.Pipeline): sklearn pipeline
        """
        numeric_transformer = Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_cols),
            ],
        )

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        )

        return pipe