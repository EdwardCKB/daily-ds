from sklearn.datasets import load_iris
import pandas as pd

def load_data() -> pd.DataFrame:
    iris = load_iris()
    return pd.DataFrame(data=iris.data, columns=iris.feature_names)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df

def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()

def main():
    df = load_data()
    df_cleaned = clean_data(df)
    summary_stats = compute_summary_stats(df_cleaned)
    print(summary_stats)

if __name__ == "__main__":
    main()