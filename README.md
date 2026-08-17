# Daily-ds

Personal daily practice repository for Data Science, Machine Learning, and AI — structured by topic, built with clean coding habits.

---

## Repository Structure

```
daily-ds/
├── ml/                    # Machine Learning
│   ├── clean-code/        # Implementations (clean coding practices)
│   ├── data/              # Datasets
│   └── plots/             # Generated visualizations
├── sql/                   # SQL (coming soon)
└── deep-learning/         # Deep Learning (coming soon)
```

---

## Machine Learning

### Algorithms Covered

| Category | Algorithm | Dataset | Script |
|---|---|---|---|
| Classification | Logistic Regression | Titanic | `titanic_logistic_regression.py` |
| Classification | KNN | Titanic | `titanic_knn.py` |
| Classification | Naive Bayes | Titanic | `titanic_naive_bayes.py` |
| Classification | LDA | Titanic | `titanic_lda.py` |
| Classification | SVM | Titanic | `titanic_svm.py` |
| Classification | Random Forest | Titanic | `titanic_random_forest.py` |
| Classification | XGBoost | Titanic | `titanic_xgboost.py` |
| Regression | Ridge & Lasso | Insurance | `insurance_regularization.py` |
| Clustering | K-Means / Hierarchical / DBSCAN | Mall Customers | `Mall_Customers_clustering.py` |
| Dimensionality Reduction | PCA | Mall Customers | `Mall_Customers_clustering.py` |

### Datasets

| Dataset | File | Task |
|---|---|---|
| Titanic | `train.csv` | Binary classification (survival prediction) |
| Mall Customers | `Mall_Customers.csv` | Unsupervised clustering (customer segmentation) |
| Insurance | `insurance.csv` | Regression (charge prediction) |

### Running Scripts

Run from the `ml/` directory:

```bash
cd ml/
python clean-code/titanic_logistic_regression.py
python clean-code/Mall_Customers_clustering.py
```

Generated plots are saved to `ml/plots/`.

---

## Roadmap

- [x] Machine Learning — classification, regression, clustering, PCA
- [ ] SQL
- [ ] Deep Learning
- [ ] NLP
- [ ] Computer Vision

---

## Tech Stack

- Python 3
- scikit-learn · XGBoost
- pandas · numpy
- matplotlib · seaborn
