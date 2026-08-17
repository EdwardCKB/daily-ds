from day4_titanic_cleaning import load_data, clean_data

def test_load_data():
    """test if data can be loaded"""
    df = load_data("ml/data/train.csv")
    assert df is not None

def test_no_nulls():
    """test for no null values in any columns"""
    df = load_data("ml/data/train.csv")
    df = clean_data(df)
    assert df.isnull().sum().sum() == 0

def test_no_cabin():
    """check Cabin column is gone"""
    df = load_data("ml/data/train.csv")
    df = clean_data(df)
    assert "Cabin" not in df.columns

def test_row_count():
    """test number of rows after cleaning"""
    df = load_data("ml/data/train.csv")
    df = clean_data(df)
    assert df.shape[0] < 891