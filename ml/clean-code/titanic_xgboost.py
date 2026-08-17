from day4_titanic_cleaning import load_data, clean_data
from titanic_logistic_regression import train_test_split_data, matrix_calculation
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

def xgboost_pipeline():
    """Building xgboost pipeline"""
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
        ('cat', OneHotEncoder(drop='first'), ['Sex', 'Embarked'])
    ])
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', XGBClassifier(random_state=42))
    ])
    return pipeline

def hyperparameter_tunning(pipeline, X_train, y_train):
    """Construcing hyperparameter tunning to get model's best hyperparamters"""
    param_grid = {
        'model__n_estimators':[50, 100, 150, 200],
        'model__max_depth':[4, 6, 8],
        'model__min_child_weight': [1, 3, 5],
        'model__learning_rate':[0.01, 0.05, 0.1]
    }
    grid_sesarch = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1')
    grid_sesarch.fit(X_train, y_train)
    return grid_sesarch.best_estimator_    

def main():
    df = load_data("ml/data/train.csv")
    df = clean_data(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df, 0.2)
    pipeline = xgboost_pipeline()
    best_model = hyperparameter_tunning(pipeline, X_train, y_train)
    pred = best_model.predict(X_test)
    accuracy, precision, recall, f1, matrix = matrix_calculation(y_test, pred)
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\nmatrix: {matrix}")
    print(confusion_matrix(y_test, pred, labels=best_model.classes_))
    print(best_model.named_steps['model'].get_params())

if __name__ == "__main__":
    main()
    