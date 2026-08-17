from day4_titanic_cleaning import load_data, clean_data
from titanic_logistic_regression import train_test_split_data, matrix_calculation, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import pandas as pd

def build_knn_pipeline():
    """Building Knn pipeline"""
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
        ('cat', OneHotEncoder(drop='first'), ['Sex', 'Embarked'])
    ])
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', KNeighborsClassifier())
    ])
    return pipeline

def hyperparameter_tunning(pipeline, X_train, y_train):
    """Finding best k for KNN model"""
    params_grid ={
        'model__n_neighbors':[3, 5, 7 ,9 ,11]
    }
    grid_search = GridSearchCV(pipeline, params_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def main():
    df = load_data("ml/data/train.csv")
    df = clean_data(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df, 0.2)
    knn_pipeline = build_knn_pipeline()
    best_knn_pipeline = hyperparameter_tunning(knn_pipeline, X_train, y_train)
    pred = best_knn_pipeline.predict(X_test)
    accuracy, precision, recall, f1, matrix = matrix_calculation(y_test, pred)
    print(f"Best k value: k={best_knn_pipeline.named_steps['model'].get_params()['n_neighbors']}")
    print(f"Metrics for KNN model\nAccuracy:{accuracy}\nPrecision:{precision}\nrecall:{recall}\nF1:{f1}\nMatrix:{matrix}")
if __name__ == "__main__":
    main()
    

