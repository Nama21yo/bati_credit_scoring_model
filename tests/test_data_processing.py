import pandas as pd
import pytest
from src.data_processing import create_aggregate_features, extract_time_features


def test_create_aggregate_features():
    """Test aggregate feature creation."""
    data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, 200, 150],
        'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-03']
    })
    result = create_aggregate_features(data)
    assert len(result) == 2
    assert result.loc[result['CustomerId'] ==
                      'C1', 'total_amount'].iloc[0] == 300
    assert result.loc[result['CustomerId'] == 'C1', 'txn_count'].iloc[0] == 2


def test_extract_time_features():
    """Test extraction of time-based features."""
    data = pd.DataFrame({
        'CustomerId': ['C1'],
        'Amount': [100],
        'TransactionStartTime': ['2023-01-01 14:30:00']
    })
    result = extract_time_features(data)
    assert result['txn_hour'].iloc[0] == 14
    assert result['txn_day'].iloc[0] == 1
    assert result['txn_month'].iloc[0] == 1
