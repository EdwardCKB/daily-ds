from day4_titanic_cleaning import load_data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd


def split_train_test(df:pd.DataFrame, test_size:float):
    """Split my dataset into X, y, train and test"""
    y = df["charges"]
    X = df.drop(columns=["charges"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, y_train, X_test, y_test

def build_ridge_pipeline():
    """Building Ridge pipeline"""
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['age', 'bmi', 'children']),
        ('cat', OneHotEncoder(drop='first'), ['sex', 'smoker', 'region'])
    ])
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', Ridge(random_state=42)) #Ridge = Linear Regression + L2(Ridge penalty) built in
    ])
    return pipeline

def build_lasso_pipeline():
    """Building Lasso pipeline"""
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['age', 'bmi', 'children']),
        ('cat', OneHotEncoder(drop='first'), ['sex', 'smoker', 'region'])
    ])
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', Lasso(random_state=42)) #Lasso = Linear Regression + L1 penalty
    ])
    return pipeline 

def tune_lasso(pipeline, X_train, y_train):
    """Construcing hyperparameter tunning to get model's best hyperparamters"""
    param_grid = {
        'model__alpha': [0.1, 1, 10, 50, 100, 500]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_  

def evalutate_regression(y_test, pred):
    MAE = mean_absolute_error(y_test, pred)
    MSE = mean_squared_error(y_test, pred)
    R2 = r2_score(y_test, pred)
    return MAE, MSE, R2


def main():
    df = load_data("data/insurance.csv")
    #print(df.info())
    #print(df.nunique())
    #print(df.shape)
    X_train, y_train, X_test, y_test = split_train_test(df, 0.2)
    ridge_pipeline = build_ridge_pipeline()
    ridge_pipeline.fit(X_train, y_train)
    ridge_pred = ridge_pipeline.predict(X_test)

    lasso_pipeline = build_lasso_pipeline()
    best_lasso_model = tune_lasso(lasso_pipeline, X_train, y_train)
    lasso_pred = best_lasso_model.predict(X_test)

    r_MAE, r_MSE, r_R2 = evalutate_regression(y_test, ridge_pred)
    print(f"Ridge model metrics:\nMAE: {r_MAE}\nMSE: {r_MSE}\nR2: {r_R2}")
    l_MAE, l_MSE, l_R2 = evalutate_regression(y_test, lasso_pred)
    print("=======================")
    print(f"Lasso model metrics:\nMAE: {l_MAE}\nMSE: {l_MSE}\nR2: {l_R2}")

    feature_names = ridge_pipeline.named_steps['preprocess'].get_feature_names_out()
    ridge_coefs = ridge_pipeline.named_steps['model'].coef_
    lasso_coefs = best_lasso_model.named_steps['model'].coef_

    comparison = pd.DataFrame({
        'feature': feature_names,
        'ridge': ridge_coefs,
        'lasso': lasso_coefs
    })
    print(comparison)
    print(best_lasso_model.named_steps['model'].alpha)


if __name__ == "__main__":
    main()
    