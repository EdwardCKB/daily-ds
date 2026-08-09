from day4_titanic_cleaning import load_data, clean_data
from titanic_logistic_regression import train_test_split_data, matrix_calculation, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import pandas as pd

def build_svm_pipeline():
    """Building SVM pipeline"""
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
        ('cat', OneHotEncoder(drop='first'), ['Sex', 'Embarked'])
    ])
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', SVC())
    ])
    return pipeline

def hyperparameter_tunning(pipeline, X_train, y_train):
    params_grid = {
        'model__C':[0.1, 1, 10, 100],
        'model__kernel':['linear', 'rbf']
    }
    grid_search = GridSearchCV(pipeline, params_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def main():
    df = load_data("data/train.csv")
    df = clean_data(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df, 0.2)
    svm_pipeline = build_svm_pipeline()
    best_svm_pipeline = hyperparameter_tunning(svm_pipeline, X_train, y_train)
    pred = best_svm_pipeline.predict(X_test)
    accuracy, precision, recall, f1, matrix = matrix_calculation(y_test, pred)
    print(f"Best kernel: k={best_svm_pipeline.named_steps['model'].get_params()['kernel']}")
    print(f"Metrics for KNN model\nAccuracy:{accuracy}\nPrecision:{precision}\nrecall:{recall}\nF1:{f1}\nMatrix:{matrix}")

if __name__ == "__main__":
    main()
    

