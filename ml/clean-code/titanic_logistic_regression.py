from day4_titanic_cleaning import load_data, clean_data, encoding
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

def matrix_calculation(y_test, pred):
    """Calculate model performance metrics"""
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    matrix = confusion_matrix(y_test, pred)
    return accuracy, precision, recall, f1, matrix

def main():
    df = load_data("ml/data/train.csv")
    df = clean_data(df)
    df = encoding(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df, 0.2)
    feature_names = X_train.columns
    X_train, X_test = scaling(X_train, X_test)
    model = logistic_regression_model(X_train, y_train)
    print(pd.Series(model.coef_[0], index=feature_names))
    pred = model.predict(X_test)
    accuracy, precision, recall, f1, matrix = matrix_calculation(y_test, pred)
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\nmatrix: {matrix}")

if __name__ == "__main__":
    main()