from day4_titanic_cleaning import load_data, clean_data
from titanic_logistic_regression import train_test_split_data, matrix_calculation, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

def build_naivebayes_pipeline():
    """Build Naive Bayes pipeline"""
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
        ('cat', OneHotEncoder(), ['Sex', 'Embarked'])
    ])
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', GaussianNB())
    ])
    return pipeline

def main():
    df = load_data('data/train.csv')
    df = clean_data(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df, 0.2)
    nb_pipeline = build_naivebayes_pipeline()
    nb_pipeline.fit(X_train, y_train)
    pred = nb_pipeline.predict(X_test)
    accuracy, precision, recall, f1, matrix = matrix_calculation(y_test, pred)
    print(f"Metrics for Naive Bayes model\nAccuracy:{accuracy}\nPrecision:{precision}\nrecall:{recall}\nF1:{f1}\nMatrix:{matrix}")

if __name__ == "__main__":
    main()
