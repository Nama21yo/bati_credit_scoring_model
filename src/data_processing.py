import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import kagglehub
from kagglehub import KaggleDatasetAdapter
from xverse.transformer import WOE

def load_data():
    """Load the raw dataset from Kaggle."""
    file_path = "training.csv"
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "atwine/xente-challenge",
        file_path,
    )
    return df
