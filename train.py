import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def train_and_evaluate():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split 80% train and 20% test, stratifying to ensure equal class proportions
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
        'SVM (SVC)': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gaussian Naive Bayes': GaussianNB()
    }
    
    metrics = {}
    trained_models = {}
    
    print("\n--- Training and Evaluating ML Models (80/20 Split) ---")
    print(f"{'Model':<25} | {'Train Acc':<9} | {'Test Acc':<8} | {'Train Err':<9} | {'Test Err':<8} | {'Fitting Status':<20}")
    print("-" * 90)
    
    best_test_acc = -1
    best_model_name = None
    best_model_obj = None
    
    for name, model in models.items():
        # Fit model
        model.fit(train_X, train_y)
        trained_models[name] = model
        
        # Predict
        train_pred = model.predict(train_X)
        test_pred = model.predict(test_X)
        
        # Accuracies
        train_acc = accuracy_score(train_y, train_pred)
        test_acc = accuracy_score(test_y, test_pred)
        
        # Error Rates
        train_err = 1.0 - train_acc
        test_err = 1.0 - test_acc
        
        # Fitting status check
        if train_acc - test_acc > 0.05:
            status = 'Slightly Overfitting'
        elif train_acc < 0.85 and test_acc < 0.85:
            status = 'Underfitting'
        else:
            status = 'Well-fitted (Good)'
            
        metrics[name] = {
            'train_acc': float(train_acc),
            'test_acc': float(test_acc),
            'train_err': float(train_err),
            'test_err': float(test_err),
            'status': status
        }
        
        print(f"{name:<25} | {train_acc:.4f}    | {test_acc:.4f}   | {train_err:.4f}    | {test_err:.4f}   | {status:<20}")
        
        # Select best model: highest test accuracy
        # Tie breaker: simpler model (e.g. SVM or Logistic Regression or Random Forest)
        is_better = False
        if test_acc > best_test_acc:
            is_better = True
        elif test_acc == best_test_acc:
            # Tie breaking priority
            priority = ['SVM (SVC)', 'Logistic Regression', 'Random Forest', 'KNN (k=3)', 'Gaussian Naive Bayes', 'Decision Tree']
            if name in priority and best_model_name in priority:
                if priority.index(name) < priority.index(best_model_name):
                    is_better = True
            elif name in priority:
                is_better = True
                
        if is_better:
            best_test_acc = test_acc
            best_model_name = name
            best_model_obj = model

    print(f"\nBest Model: {best_model_name} (Test Accuracy: {best_test_acc:.4f})")
    
    # Save consolidated models and metrics
    consolidated_data = {
        'models': trained_models,
        'metrics': metrics,
        'best_model_name': best_model_name,
        'feature_names': iris.feature_names,
        'target_names': list(iris.target_names)
    }
    
    with open("models.pkl", "wb") as f:
        pickle.dump(consolidated_data, f)
    print("Saved all models and metrics to 'models.pkl'")
    
    # Save best model to Data.pkl for direct load in app
    with open("Data.pkl", "wb") as f:
        pickle.dump(best_model_obj, f)
    print(f"Saved the best model ({best_model_name}) to 'Data.pkl'")

if __name__ == "__main__":
    train_and_evaluate()
