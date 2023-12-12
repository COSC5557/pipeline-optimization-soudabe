import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC


def split_data(df, test_size=0.2, random_state=42):
    return train_test_split(df, test_size=test_size, random_state=random_state)


def separate_features_target(df, target_column):
    features = df.drop([target_column], axis=1)
    target = df[target_column].copy()
    return features, target


def preprocess_pipeline():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])


def nested_cross_validation(X, y, classifier, param_grid):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    nested_scores = []
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = Pipeline([
            ('preprocessor', preprocess_pipeline()),
            ('classifier', classifier)
        ])

        random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=inner_cv, random_state=42)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        nested_score = best_model.score(X_test, y_test)
        nested_scores.append(nested_score)

    return np.mean(nested_scores)


def plot_comparison(labels, nested_cv_scores):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, nested_cv_scores, color='tab:blue')
    plt.xlabel('Classifiers')
    plt.ylabel('Nested CV Accuracy')
    plt.title('Nested Cross-Validation Accuracy Comparison')
    plt.show()


if __name__ == "__main__":
    data_frame = pd.read_csv("winequality-white.csv", sep=";")
    X = data_frame.drop('quality', axis=1)
    y = data_frame['quality']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    classifiers = {
        'SVM': (SVC(), {
            'classifier__C': [0.001, 0.01, 10, 100],
            'classifier__gamma': [0.1, 0.01, 0.001, 0.0001],
            'classifier__kernel': ['linear', 'rbf', 'poly']
        }),
        'Random Forest': (RandomForestClassifier(), {
            'classifier__n_estimators': [5, 20, 50, 100],
            'classifier__max_depth': [int(x) for x in np.linspace(10, 120, num=12)],
            'classifier__max_features': ['auto', 'sqrt'],
            'classifier__min_samples_leaf': [1, 3, 4],
            'classifier__min_samples_split': [2, 6, 10],
            'classifier__bootstrap': [True, False],
        }),
        'Logistic Regression': (LogisticRegression(), {
            'classifier__penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'classifier__C': np.logspace(-4, 4, 20),
            'classifier__solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
            'classifier__max_iter': [100, 1000, 2500, 5000]
        }),
        'K-Nearest Neighbors': (KNeighborsClassifier(), {
            'classifier__n_neighbors': np.arange(2, 30, 1),
            'classifier__weights': ['uniform', 'distance']
        })
    }

    results = {}

    for clf_name, (classifier, param_grid) in classifiers.items():
        nested_cv_score = nested_cross_validation(X, y, classifier, param_grid)
        results[clf_name] = nested_cv_score
        print(f"Classifier: {clf_name}")
        print(f"Nested CV Accuracy: {nested_cv_score:.4f}")
        print("---------------------------------------------------------")

    labels = list(results.keys())
    nested_cv_scores = list(results.values())

    plot_comparison(labels, nested_cv_scores)
