import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import auc

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent to optimize weights
def gradient_descent(X, y, weights, learning_rate, iterations):
    cost_history = []
    for i in range(iterations):
        m = len(y)
        predictions = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (predictions - y)) / m
        weights -= learning_rate * gradient  # Update rule

        # Clip predictions to avoid log(0) and log(1)
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        
        cost = - (1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        cost_history.append(cost)

    return weights, cost_history

# Function to calculate TPR and FPR for each fold
def compute_tpr_fpr(y_true, y_pred, thresholds):
    tpr_list = []
    fpr_list = []
    for threshold in thresholds:
        predictions = (y_pred >= threshold).astype(int)
        
        # True Positive (TP), False Positive (FP), False Negative (FN), True Negative (TN)
        TP = np.sum((predictions == 1) & (y_true == 1))
        FP = np.sum((predictions == 1) & (y_true == 0))
        FN = np.sum((predictions == 0) & (y_true == 1))
        TN = np.sum((predictions == 0) & (y_true == 0))
        
        # True Positive Rate (TPR) = TP / (TP + FN)
        # False Positive Rate (FPR) = FP / (FP + TN)
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        tpr_list.append(TPR)
        fpr_list.append(FPR)
    
    return tpr_list, fpr_list

# K-fold Cross-Validation
def k_fold_cross_validation(X, y, k=10, learning_rate=0.01, iterations=50):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    thresholds = np.arange(0, 1.1, 0.1)  # thresholds: [0, 0.1, 0.2, ..., 1.0]
    all_tpr = []
    all_fpr = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Initialize weights to zeros
        weights = np.zeros(X_train.shape[1])
        
        # Train logistic regression using gradient descent
        weights_optimal, _ = gradient_descent(X_train, y_train, weights, learning_rate, iterations)
        
        # Predict the probabilities for the test set
        y_pred = sigmoid(np.dot(X_test, weights_optimal))
        
        # Compute TPR and FPR
        tpr, fpr = compute_tpr_fpr(y_test, y_pred, thresholds)
        
        all_tpr.append(tpr)
        all_fpr.append(fpr)
    
    # Average TPR and FPR across all folds
    avg_tpr = np.mean(all_tpr, axis=0)
    avg_fpr = np.mean(all_fpr, axis=0)
    
    return avg_tpr, avg_fpr, thresholds

# Plot ROC curve and calculate AUC
def plot_roc_curve(avg_tpr, avg_fpr):
    plt.figure(figsize=(8, 6))
    plt.plot(avg_fpr, avg_tpr, color='b', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    
    # Calculate AUC
    roc_auc = auc(avg_fpr, avg_tpr)

    return roc_auc

# Main function
def main():
    # Data collection
    data = pd.read_csv('MNIST_CV.csv')

    # Data preprocessing
    X = data.drop(columns=['label']).values
    y = data['label'].values
    
    # Normalize data
    X = X / 255.0
    y = (y - y.min()) / (y.max() - y.min())
    
    # Add intercept
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Run K-fold cross-validation
    avg_tpr, avg_fpr, thresholds = k_fold_cross_validation(X, y, k=10, iterations=1000)
    
    # Averaged TPR and FPR table
    print("\nAveraged TPR and FPR at Thresholds:")
    print("Threshold\tTPR\t\tFPR")
    for threshold, tpr, fpr in zip(thresholds, avg_tpr, avg_fpr):
        print(f"{threshold:.1f}\t\t{tpr:.4f}\t\t{fpr:.4f}")
    
    # Plot ROC curve and calculate AUC
    auc_score = plot_roc_curve(avg_tpr, avg_fpr)
    print(f"\nFinal AUC: {auc_score:.4f}")

# Run the main function
if __name__ == '__main__':
    auc_score = main()