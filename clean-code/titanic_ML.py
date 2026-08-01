from day4_titanic_cleaning import load_data, clean_data, encoding
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd


def train_test_split_data(df: pd.DataFrame, desired_test_size: float):
    """Split Data into train and test sets"""
    y =df['Survived']
    X = df.drop(columns = ['Survived', 'Ticket', 'Name'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = desired_test_size, random_state = 42, stratify = y)
    return X_train, X_test, y_train, y_test

def scaling(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale features to have mean 0 and standard deviation 1"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
    
def logistic_regression_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Building Logistic Regression model"""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model
    
def main():
    df = load_data("data/train.csv")
    df = clean_data(df)
    df = encoding(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df, 0.2)
    feature_names = X_train.columns
    X_train, X_test = scaling(X_train, X_test)
    model = logistic_regression_model(X_train, y_train)
    print(pd.Series(model.coef_[0], index=feature_names))
    pred = model.predict(X_test)
    #print(pred)

if __name__ == "__main__":
    main()