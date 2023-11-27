import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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


def preprocess_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


def create_pipeline(preprocessor, classifier):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])


def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    end_time = time.time()
    accuracy = pipeline.score(X_test, y_test)
    return accuracy, end_time - start_time


def hyperparameter_tuning(pipeline, param_grid, X_train, y_train, X_test, y_test):
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=5, random_state=42)

    start_time = time.time()
    random_search.fit(X_train, y_train)
    end_time = time.time()

    best_params = random_search.best_params_
    accuracy = random_search.score(X_test, y_test)

    return best_params, accuracy, end_time - start_time


def plot_comparison(labels, accuracy_scores, training_times):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.bar(labels, accuracy_scores, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Training Time (seconds)', color=color)
    ax2.plot(labels, training_times, marker='o', linestyle='-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Accuracy and Training Time Comparison')
    plt.show()


if __name__ == "__main__":
    data_frame = pd.read_csv("winequality-white.csv", sep=";")
    train_set, test_set = split_data(data_frame)
    X = data_frame.drop('quality', axis=1)
    X_train, y_train = separate_features_target(train_set, target_column="quality")
    X_test, y_test = separate_features_target(test_set, target_column="quality")

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = preprocess_pipeline(numeric_features, categorical_features)

    classifiers = {
        'SVM': (SVC(), 'SVM', {
            'classifier__C': [0.001, 0.01, 10, 100],
            'classifier__gamma': [0.1, 0.01, 0.001, 0.0001],
            'classifier__kernel': ['linear', 'rbf', 'poly']
        }),
        'Random Forest': (RandomForestClassifier(), 'RF', {
            'classifier__n_estimators': [5,20,50,100],
            'classifier__max_depth': [int(x) for x in np.linspace(10, 120, num = 12)],
            'classifier__max_features': ['auto', 'sqrt'],
            'classifier__min_samples_leaf': [1, 3, 4],
            'classifier__min_samples_split': [2, 6, 10],
            'classifier__bootstrap': [True, False],
        }),
        'Logistic Regression': (LogisticRegression(), 'LR', {
            'classifier__penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'classifier__C': np.logspace(-4, 4, 20),
            'classifier__solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
            'classifier__max_iter': [100, 1000, 2500, 5000]
        }),
        'K-Nearest Neighbors': (KNeighborsClassifier(), 'KNN', {
            'classifier__n_neighbors': np.arange(2, 30, 1),
            'classifier__weights': ['uniform', 'distance']
        })
    }

    results = {}

    for clf_name, (classifier, label, param_grid) in classifiers.items():
        pipeline = create_pipeline(preprocessor, classifier)
        simple_accuracy, simple_time = evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test)
        best_params, optimized_accuracy, optimized_time = hyperparameter_tuning(pipeline, param_grid, X_train, y_train,
                                                                                X_test, y_test)
        results[clf_name] = {
            'label': label,
            'best_params': best_params,
            'simple_accuracy': simple_accuracy,
            'simple_time': simple_time,
            'optimized_accuracy': optimized_accuracy,
            'optimized_time': optimized_time
        }

    for clf_name, result in results.items():
        print(f"Classifier: {clf_name}")
        print(f"Time for fitting the simple pipeline: {result['simple_time']:.4f} seconds")
        print(f"Accuracy without hyperparameter optimization: {result['simple_accuracy']:.4f}")
        print(f"Best Hyperparameters: {result['best_params']}")
        print(f"Time for fitting the pipeline with hyperparameter optimization: {result['optimized_time']:.4f} seconds")
        print(f"Accuracy with hyperparameter optimization: {result['optimized_accuracy']:.4f}")
        print("---------------------------------------------------------")

    labels = [result['label'] for result in results.values()]
    accuracy_scores = [result['optimized_accuracy'] for result in results.values()]
    training_times = [result['optimized_time'] for result in results.values()]

    plot_comparison(labels, accuracy_scores, training_times)