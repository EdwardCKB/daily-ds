import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load raw Titanic data from a CSV file."""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Titanic dataset: fill missing ages, drop Cabin, drop rows missing Embarked."""
    mean_age = df['Age'].mean()
    df['Age'] = df['Age'].fillna(mean_age)
    df = df.drop(columns=['Cabin'])
    df = df.dropna(subset=['Embarked'])
    return df

def encoding(df:pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables"""
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first= True)
    return df

def main():
    df = load_data("data/train.csv")
    df = clean_data(df)
    df = encoding(df)

if __name__ == "__main__":
    main()