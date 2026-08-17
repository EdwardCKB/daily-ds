from day4_titanic_cleaning import load_data, clean_data
from titanic_logistic_regression import train_test_split_data, matrix_calculation, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def plot_svm_boundary(best_svm_pipeline, df):
    """Visualize decision boundary using Sex and Fare (2 features, so it can be drawn)"""
    best_C = best_svm_pipeline.named_steps['model'].get_params()['C']
    best_kernel = best_svm_pipeline.named_steps['model'].get_params()['kernel']

    df_plot = df.copy()
    df_plot['Sex_encoded'] = df_plot['Sex'].map({'male': 0, 'female': 1})

    X = df_plot[['Sex_encoded', 'Fare']].values
    y = df_plot['Survived'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel=best_kernel, C=best_C)
    model.fit(X_scaled, y)

    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=['#f4b6b6', '#a9c9f0'])
    plt.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], c='red', s=20, label='Died')
    plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], c='blue', s=20, label='Survived')
    plt.xlabel('Sex (scaled: male=left cluster, female=right cluster)')
    plt.ylabel('Fare (scaled)')
    plt.title(f'SVM boundary (kernel={best_kernel}, C={best_C}) — Sex vs Fare')
    plt.legend()
    plt.savefig('ml/plots/svm_boundary_sex.png', dpi=150)
    plt.show()
    
def main():
    df = load_data("ml/data/train.csv")
    df = clean_data(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df, 0.2)
    svm_pipeline = build_svm_pipeline()
    best_svm_pipeline = hyperparameter_tunning(svm_pipeline, X_train, y_train)
    pred = best_svm_pipeline.predict(X_test)
    accuracy, precision, recall, f1, matrix = matrix_calculation(y_test, pred)
    print(f"Best kernel: k={best_svm_pipeline.named_steps['model'].get_params()['kernel']}")
    print(f"Metrics for KNN model\nAccuracy:{accuracy}\nPrecision:{precision}\nrecall:{recall}\nF1:{f1}\nMatrix:{matrix}")
    plot_svm_boundary(best_svm_pipeline, df)

if __name__ == "__main__":
    main()