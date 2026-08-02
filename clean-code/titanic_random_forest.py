from day4_titanic_cleaning import load_data, clean_data
from titanic_logistic_regression import train_test_split_data, matrix_calculation
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

def build_pipeline():
    """Building model pipeline"""
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
        ('cat', OneHotEncoder(drop='first'), ['Sex', 'Embarked'])
    ])
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', RandomForestClassifier())
    ])
    return pipeline

def main():
    df = load_data("data/train.csv")
    df = clean_data(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df, 0.2)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    accuracy, precision, recall, f1, matrix = matrix_calculation(y_test, pred)
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\nmatrix: {matrix}")

if __name__ == "__main__":
    main()
    

