from day4_titanic_cleaning import load_data, clean_data, encoding
from sklearn.model_selection import train_test_split
import pandas as pd


def train_test_split_data(df: pd.DataFrame, desired_test_size: float):
    """Split Data into train and test sets"""
    y =df['Survived']
    X = df.drop(columns = ['Survived'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = desired_test_size, random_state = 42)
    return X_train, X_test, y_train, y_test

def main():
    df = load_data("data/train.csv")
    df = clean_data(df)
    df = encoding(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df, 0.2)
    print(f"Train set size: {len(X_train)} \nTest set size {len(X_test)}" )

if __name__ == "__main__":
    main()