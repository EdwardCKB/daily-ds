from sklearn.datasets import load_wine
import pandas as pd

def load_data() -> pd.DataFrame:
    """load data"""
    wine = load_wine()
    df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """clean df"""
    return df

def filter_by_class(df: pd.DataFrame, target_class: int) -> pd.DataFrame:
    """filter the choice of target class wine you want"""
    return df[df["target"] == target_class]

def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """look at data summary"""
    return df.describe()

def main():
    df = load_data()
    df1 = clean_data(df)
    df2 = filter_by_class(df1, 0)
    print(compute_summary_stats(df2))

if __name__ == "__main__":
    main()