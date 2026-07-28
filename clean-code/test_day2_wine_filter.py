from day2_wine_filter import load_data, clean_data, filter_by_class, compute_summary_stats
from sklearn.datasets import load_wine
def test_load_data():
    df = load_data()
    assert df is not None

def test_filter_by_class():
    df = load_data()
    filtered = filter_by_class(df, 0)
    assert filtered["target"].unique() == [0]

def test_no_missing_values():
    df = load_data()
    assert df.isnull().sum().sum() == 0

def test_alcohol_is_numeric():
    df = load_data()
    assert df["alcohol"].dtype == "float64"
