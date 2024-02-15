import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from typing import Tuple


def create_embeddings(dataframe: pd.DataFrame, feature: str) -> np.ndarray:
    """
    Create embeddings for a specific feature using a multilingual model
    Args:
        dataframe (pd.DataFrame): data to encode
        feature (str): feature to encode
    Returns:
        embeddings (np.ndarray): array with embeddings
    """

    # model to create embeddings
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # embeddings
    dataframe = dataframe.loc[:, ["ISBN", feature]].drop_duplicates()
    embeddings = model.encode(dataframe[feature].tolist(), normalize_embeddings=True)

    return embeddings


def reduce_dimensionality(embeddings: np.ndarray, components: Tuple[float, int]) -> Tuple[np.ndarray, PCA]:
    """
    Apply dimensionality reduction using PCA
    Args:
        embeddings (list): list with embeddings, the PCA will be fitted to the first embedding in the list
        components (Tuple[float, int]): number of components for PCA
    Returns:
        Tuple[np.ndarray, PCA]: embeddings reduced and fitted PCA
    """

    # reduce dimensionality with PCA
    pca = PCA(n_components=components, random_state=42)
    train_embeddings = pca.fit_transform(embeddings)

    return train_embeddings, pca


def add_embeddings_to_df(dataframe: pd.DataFrame, embeddings: np.ndarray, feature: str) -> pd.DataFrame:
    """
    Add embeddings to dataframe
    Args:
        dataframe (pd.DataFrame): data
        embeddings (np.ndarray): embeddings
        feature (str): feature to encode
    Returns:
        dataframe (pd.DataFrame): data with embeddings
    """

    # join embeddings to dfs
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.columns = [f"{feature}_{i}" for i in embeddings_df.columns]
    book_df = dataframe.loc[:, ["ISBN", feature]].drop_duplicates()
    book_df = pd.merge(book_df.reset_index(drop=True), embeddings_df, left_index=True, right_index=True)
    dataframe = pd.merge(dataframe, book_df, on=["ISBN", feature])

    return dataframe


def define_pipeline(
        numerical_cols: list,
    ) -> Pipeline:
        """
        Defines data transformation pipeline
        Args:
            numerical_cols (list): list of numerical features to be imputed with median
        Returns:
            pipe (sk.Pipeline): sklearn pipeline
        """
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
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
            ]
        )

        return pipe
