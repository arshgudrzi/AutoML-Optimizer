#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import warnings
warnings.filterwarnings("ignore")

def load_data(csv_file):
    """Load CSV data into a DataFrame."""
    try:
        df = pd.read_csv(csv_file)
        print(f"[+] Data loaded from {csv_file}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[-] Error loading CSV: {e}")
        exit(1)

def preprocess_data(df, feature_cols, target_col):
    """Preprocess the data: handle missing values and encoding."""
    # Drop rows with missing values (customize as needed)
    df = df.dropna(subset=feature_cols + [target_col])
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # If target is non-numeric, encode it (for classification tasks)
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print("[*] Target variable encoded for classification.")
        task = 'classification'
    else:
        task = 'regression'
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, task

def create_tf_model(input_dim, task='classification'):
    """Create a simple TensorFlow DNN model."""
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_dim=input_dim))
    model.add(layers.Dense(32, activation='relu'))
    
    if task == 'classification':
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    
    return model

def evaluate_models(X, y, task='classification'):
    """Evaluate multiple models using cross-validation and select the best one."""
    models_to_evaluate = {}
    
    if task == 'classification':
        models_to_evaluate = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'DecisionTree': DecisionTreeClassifier(),
            'RandomForest': RandomForestClassifier(n_estimators=100)
        }
    else:
        models_to_evaluate = {
            'LinearRegression': LinearRegression(),
            'DecisionTree': DecisionTreeRegressor(),
            'RandomForest': RandomForestRegressor(n_estimators=100)
        }
    
    scores = {}
    for name, model in models_to_evaluate.items():
        print(f"[*] Evaluating {name}...")
        # 5-fold cross-validation; use accuracy for classification, negative MSE for regression
        cv_score = cross_val_score(model, X, y, cv=5,
                                   scoring="accuracy" if task=='classification' else "neg_mean_squared_error")
        mean_score = np.mean(cv_score)
        scores[name] = mean_score
        print(f"    Score: {mean_score:.4f}")
    
    # Evaluate TensorFlow model using a simple split, since it is not scikit-learn compatible by default
    # Here we create a wrapper for compatibility.
    tf_model = create_tf_model(X.shape[1], task)
    tf_model.fit(X, y, epochs=20, batch_size=32, verbose=0, validation_split=0.2)
    if task == 'classification':
        loss, acc = tf_model.evaluate(X, y, verbose=0)
        tf_score = acc
        print(f"[*] Evaluating TensorFlow DNN: Accuracy {acc:.4f}")
    else:
        loss, mse = tf_model.evaluate(X, y, verbose=0)
        tf_score = -mse  # negative mse for consistency with cv scoring
        print(f"[*] Evaluating TensorFlow DNN: MSE {mse:.4f}")
    
    scores['TensorFlow_DNN'] = tf_score
    
    # Select best model based on score
    if task == 'classification':
        best_model_name = max(scores, key=scores.get)
    else:
        best_model_name = max(scores, key=scores.get)
    
    print(f"\n[+] Best performing model: {best_model_name} with score {scores[best_model_name]:.4f}")
    return best_model_name, scores

def grid_search_optimization(model, X, y, task='classification'):
    """Run GridSearchCV on the given model to optimize hyperparameters."""
    print(f"[*] Running GridSearchCV for hyperparameter tuning on {model.__class__.__name__}...")
    
    if task == 'classification':
        if isinstance(model, RandomForestClassifier):
            param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 5, 10]}
        elif isinstance(model, DecisionTreeClassifier):
            param_grid = {'max_depth': [None, 5, 10, 20]}
        elif isinstance(model, LogisticRegression):
            param_grid = {'C': [0.01, 0.1, 1, 10]}
        else:
            print("[*] No grid search parameters defined for this model.")
            return model
        scoring = 'accuracy'
    else:
        if isinstance(model, RandomForestRegressor):
            param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 5, 10]}
        elif isinstance(model, DecisionTreeRegressor):
            param_grid = {'max_depth': [None, 5, 10, 20]}
        elif isinstance(model, LinearRegression):
            print("[*] LinearRegression has no hyperparameters to tune.")
            return model
        else:
            print("[*] No grid search parameters defined for this model.")
            return model
        scoring = 'neg_mean_squared_error'
    
    grid = GridSearchCV(model, param_grid, cv=5, scoring=scoring)
    grid.fit(X, y)
    print(f"[+] Best parameters: {grid.best_params_}")
    print(f"[+] Best score from grid search: {grid.best_score_:.4f}")
    return grid.best_estimator_

def plot_results(scores, task='classification'):
    """Plot model performance scores for visual comparison."""
    models = list(scores.keys())
    performance = list(scores.values())
    
    plt.figure(figsize=(8, 5))
    plt.bar(models, performance, color='skyblue')
    plt.title("Model Performance Comparison")
    ylabel = "Accuracy" if task=='classification' else "Negative MSE"
    plt.ylabel(ylabel)
    plt.xlabel("Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # User input for CSV path and columns
    csv_file = input("Enter the path to the CSV file: ").strip()
    df = load_data(csv_file)
    print("\nAvailable columns:")
    print(df.columns.tolist())
    
    feature_cols = input("Enter comma-separated feature column names: ").strip().split(',')
    feature_cols = [col.strip() for col in feature_cols]
    target_col = input("Enter the target column name: ").strip()
    
    # Preprocess data
    X, y, task = preprocess_data(df, feature_cols, target_col)
    
    # Split for final testing after model selection
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Evaluate different models
    best_model_name, scores = evaluate_models(X_train, y_train, task)
    plot_results(scores, task)
    
    # Choose a model instance based on best_model_name for grid search tuning
    model_dict = {}
    if task == 'classification':
        model_dict = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'DecisionTree': DecisionTreeClassifier(),
            'RandomForest': RandomForestClassifier(n_estimators=100),
            'TensorFlow_DNN': None  # TensorFlow model is handled separately
        }
    else:
        model_dict = {
            'LinearRegression': LinearRegression(),
            'DecisionTree': DecisionTreeRegressor(),
            'RandomForest': RandomForestRegressor(n_estimators=100),
            'TensorFlow_DNN': None
        }
    
    if best_model_name == 'TensorFlow_DNN':
        print("[*] Skipping grid search for TensorFlow DNN. Consider manual tuning if needed.")
        final_model = create_tf_model(X.shape[1], task)
        final_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
        # Evaluate on test set
        if task == 'classification':
            loss, acc = final_model.evaluate(X_test, y_test, verbose=0)
            print(f"[+] Final TensorFlow DNN Test Accuracy: {acc:.4f}")
        else:
            loss, mse = final_model.evaluate(X_test, y_test, verbose=0)
            print(f"[+] Final TensorFlow DNN Test MSE: {mse:.4f}")
    else:
        base_model = model_dict[best_model_name]
        # Hyperparameter tuning
        optimized_model = grid_search_optimization(base_model, X_train, y_train, task)
        # Final evaluation on test set
        optimized_model.fit(X_train, y_train)
        predictions = optimized_model.predict(X_test)
        if task == 'classification':
            acc = accuracy_score(y_test, predictions)
            print(f"[+] Final Test Accuracy: {acc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, predictions))
        else:
            mse = mean_squared_error(y_test, predictions)
            print(f"[+] Final Test MSE: {mse:.4f}")
    
if __name__ == '__main__':
    main()
