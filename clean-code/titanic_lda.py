from day4_titanic_cleaning import load_data, clean_data
from titanic_logistic_regression import train_test_split_data, matrix_calculation
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

def build_preprocessor():
    """Create (but don't fit) the ColumnTransformer"""
    # returned UNFITTED on purpose — fit once in main() on X_train only,
    # then reuse the same fitted object via .transform() on X_test.
    # earlier bug: calling this function separately for train AND test
    # created two different fitted transformers, each learning its own
    # mean/std — X_test needs to be scaled using TRAIN's stats, not its own
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
        ('cat', OneHotEncoder(drop='first'), ['Sex', 'Embarked'])
    ])


def build_lda(X_scaled, y_train):
    """Fit LDA and return the FITTED OBJECT (not just its output)"""
    # n_components capped at (num_classes - 1) — Titanic has 2 classes,
    # so 1 is the ONLY valid value here, nothing to grid search over.
    # (a 4-class target would allow up to 3, and THAT would be a real
    # n_components search question, unlike here)
    lda = LinearDiscriminantAnalysis(n_components=1)
    # .fit(), not .fit_transform() — need the OBJECT kept around so it
    # can .transform()/.predict() on test data later. earlier bug: returning
    # x_lda (just the transformed numbers) meant there was nothing with
    # a .predict() method to actually call afterward
    lda.fit(X_scaled, y_train)
    return lda


def main():
    df = load_data('data/train.csv')
    df = clean_data(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df, 0.2)

    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)  # fit + transform, train only
    X_test_scaled = preprocessor.transform(X_test)          # transform ONLY, reuses train's fit

    lda = build_lda(X_train_scaled, y_train)
    # LDA can classify directly (unlike PCA, which never sees y at all,
    # so it can never predict anything) — this works because LDA was
    # fit WITH labels from the start
    pred = lda.predict(X_test_scaled)

    accuracy, precision, recall, f1, matrix = matrix_calculation(y_test, pred)
    print(f"LDA metrics:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\nMatrix: {matrix}")

    feature_names = preprocessor.get_feature_names_out()
    # lda.coef_ shape is (1, n_features) here — ONE row because there's
    # only ONE axis (n_components=1). [0] just selects that single row,
    # NOT "axis 2" — with a 4-class target and 3 components, coef_ would
    # have 3 rows, and [0]/[1]/[2] would each be a genuinely different axis
    print(pd.Series(lda.coef_[0], index=feature_names))
    # RESULT: near-identical to Logistic Regression (0.809/0.783/0.691/0.734)
    # — expected, since LDA and Logistic Regression are closely related for
    # 2-class problems. Sex_male (-3.53) dominates again, consistent with
    # every other Titanic model — a solid cross-check, not a coincidence

if __name__ == "__main__":
    main()
    