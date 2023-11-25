import time
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
    svm_pipeline = create_pipeline(preprocessor, SVC())

    simple_accuracy, simple_time = evaluate_pipeline(svm_pipeline, X_train, y_train, X_test, y_test)

    param_grid = {
        'classifier__C': [0.001, 0.01, 10, 100],
        'classifier__gamma': [0.1, 0.01, 0.001, 0.0001],
        'classifier__kernel': ['linear', 'rbf', 'poly']
    }

    best_params, optimized_accuracy, optimized_time = hyperparameter_tuning(svm_pipeline, param_grid, X_train, y_train,
                                                                            X_test, y_test)

    print(f"Time for fitting the simple pipeline: {simple_time:.4f} seconds")
    print(f"Accuracy without hyperparameter optimization: {simple_accuracy:.4f}")
    print(f"Best Hyperparameters: {best_params}")
    print(f"Time for fitting the pipeline with hyperparameter optimization: {optimized_time:.4f} seconds")
    print(f"Accuracy with hyperparameter optimization: {optimized_accuracy:.4f}")

    plot_comparison(['Simple Pipeline', 'Optimized Pipeline'], [simple_accuracy, optimized_accuracy],
                    [simple_time, optimized_time])
