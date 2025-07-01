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


def create_aggregate_features(df):
    """Create aggregate features for each customer."""
    agg_df = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'count', 'std'],
        'TransactionStartTime': 'max'
    }).reset_index()
    agg_df.columns = ['CustomerId', 'total_amount',
                      'avg_amount', 'txn_count', 'amount_std', 'last_txn_time']
    agg_df['amount_std'] = agg_df['amount_std'].fillna(0)
    return agg_df


def extract_time_features(df):
    """Extract time-based features from TransactionStartTime."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['txn_hour'] = df['TransactionStartTime'].dt.hour
    df['txn_day'] = df['TransactionStartTime'].dt.day
    df['txn_month'] = df['TransactionStartTime'].dt.month
    df['txn_year'] = df['TransactionStartTime'].dt.year
    return df


def calculate_rfm(df):
    """Calculate RFM metrics for each customer."""
    snapshot_date = df['TransactionStartTime'].max()
    rfm = df.groupby('CustomerId').agg({
        # Recency
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',  # Frequency
        'Amount': 'sum'  # Monetary
    }).reset_index()
    rfm.columns = ['CustomerId', 'recency', 'frequency', 'monetary']
    return rfm


def cluster_customers(rfm):
    """Cluster customers based on RFM metrics and assign high-risk label."""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[['recency', 'frequency', 'monetary']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster (high recency, low frequency, low monetary)
    cluster_summary = rfm.groupby(
        'cluster')[['recency', 'frequency', 'monetary']].mean()
    # Cluster with highest recency
    high_risk_cluster = cluster_summary['recency'].idxmax()
    rfm['is_high_risk'] = rfm['cluster'].apply(
        lambda x: 1 if x == high_risk_cluster else 0)
    return rfm[['CustomerId', 'is_high_risk']]


def build_pipeline(target_col='is_high_risk'):
    """Build a preprocessing pipeline for numerical and categorical features."""
    numeric_features = ['total_amount', 'avg_amount', 'txn_count',
                        'amount_std', 'txn_hour', 'txn_day', 'txn_month']
    categorical_features = ['ProductCategory', 'ChannelId']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('woe', WOE())  # WoE requires target variable during fit
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def process_data(save=True):
    """Process unreliable data into model-ready format with proxy target."""
    df = load_data()
    df = extract_time_features(df)
    agg_df = create_aggregate_features(df)
    rfm = calculate_rfm(df)
    target_df = cluster_customers(rfm)
    processed_df = agg_df.merge(target_df, on='CustomerId')

    if save:
        processed_df.to_csv('data/processed/processed_data.csv', index=False)
    return processed_df


if __name__ == "__main__":
    process_data()
