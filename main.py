import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC


def separate_features_target(df, target_column):
    features = df.drop([target_column], axis=1)
    target = df[target_column].copy()
    return features, target


def preprocess_pipeline(numeric_strategy, categorical_strategy):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_strategy)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_strategy)),
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

        random_search = RandomizedSearchCV(classifier, param_distributions=param_grid, n_iter=10, cv=inner_cv, random_state=42)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        nested_score = best_model.score(X_test, y_test)
        nested_scores.append(nested_score)

    return np.mean(nested_scores)


def evaluate_default_performance(X, y, classifier):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(classifier, X, y, cv=outer_cv)
    return np.mean(scores)


if __name__ == "__main__":
    data_frame = pd.read_csv("winequality-white.csv", sep=";")
    X = data_frame.drop('quality', axis=1)
    y = data_frame['quality']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    classifiers = {
        'SVM': Pipeline([
            ('preprocessor', preprocess_pipeline('median', 'most_frequent')),
            ('classifier', SVC())
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', preprocess_pipeline('median', 'most_frequent')),
            ('classifier', RandomForestClassifier())
        ]),
        'K-Nearest Neighbors': Pipeline([
            ('preprocessor', preprocess_pipeline('median', 'most_frequent')),
            ('classifier', KNeighborsClassifier())
        ])
    }

    param_grids = {
        'SVM': {
            'preprocessor__num__imputer__strategy': ['median'],
            'preprocessor__cat__imputer__strategy': ['most_frequent'],
            'classifier__C': [1, 10, 100],
            'classifier__gamma': [0.01, 0.1, 'scale'],
            'classifier__kernel': ['rbf']
        },
        'Random Forest': {
            'preprocessor__num__imputer__strategy': ['mean', 'median'],
            'preprocessor__cat__imputer__strategy': ['most_frequent', 'constant'],
            'classifier__n_estimators': [5, 20, 50, 100],
            'classifier__max_depth': [int(x) for x in np.linspace(10, 120, num=12)],
            'classifier__max_features': ['auto', 'sqrt'],
            'classifier__min_samples_leaf': [1, 3, 4],
            'classifier__min_samples_split': [2, 6, 10],
            'classifier__bootstrap': [True, False],
        },
        'K-Nearest Neighbors': {
            'preprocessor__num__imputer__strategy': ['mean', 'median'],
            'preprocessor__cat__imputer__strategy': ['most_frequent', 'constant'],
            'classifier__n_neighbors': np.arange(2, 30, 1),
            'classifier__weights': ['uniform', 'distance']
        }
    }

    results_default = {}
    results_tuned = {}

    for clf_name in classifiers.keys():
        classifier = classifiers[clf_name]

        default_score = evaluate_default_performance(X, y, classifier)
        results_default[clf_name] = default_score

        param_grid = param_grids[clf_name]
        tuned_score = nested_cross_validation(X, y, classifier, param_grid)
        results_tuned[clf_name] = tuned_score

        print(f"Classifier: {clf_name}")
        print(f"Default Accuracy: {default_score:.4f}")
        print(f"Tuned Accuracy: {tuned_score:.4f}")
        print("---------------------------------------------------------")

    labels = list(results_default.keys())
    default_scores = list(results_default.values())
    tuned_scores = list(results_tuned.values())

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, default_scores, width, label='Default Accuracy')
    rects2 = ax.bar(x + width / 2, tuned_scores, width, label='Tuned Accuracy')

    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Classifier and Tuning')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()