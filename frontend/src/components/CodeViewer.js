// frontend/src/components/CodeViewer.js

import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus, vs } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './CodeViewer.css';

// Model kaynak kodları
const MODEL_SOURCE_CODES = {
  'Decision Tree': {
    title: 'Decision Tree Implementation',
    description: 'A simple decision tree classifier implementation using scikit-learn',
    code: `# Decision Tree Classifier Implementation
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

class DecisionTreeModel:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        """
        Initialize Decision Tree Classifier

        Parameters:
        - criterion: The function to measure the quality of a split
        - max_depth: The maximum depth of the tree
        - min_samples_split: The minimum number of samples required to split an internal node
        - min_samples_leaf: The minimum number of samples required to be at a leaf node
        - random_state: Random state for reproducibility
        """
        self.model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.is_trained = False

    def train(self, X_train, y_train):
        """
        Train the Decision Tree model

        Parameters:
        - X_train: Training features
        - y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"Decision Tree trained with {len(X_train)} samples")
        print(f"Tree depth: {self.model.tree_.max_depth}")
        print(f"Number of nodes: {self.model.tree_.node_count}")
        print(f"Number of leaves: {self.model.tree_.n_leaves}")

    def predict(self, X_test):
        """
        Make predictions using the trained model

        Parameters:
        - X_test: Test features

        Returns:
        - predictions: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        predictions = self.model.predict(X_test)
        return predictions

    def predict_proba(self, X_test):
        """
        Get prediction probabilities

        Parameters:
        - X_test: Test features

        Returns:
        - probabilities: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        probabilities = self.model.predict_proba(X_test)
        return probabilities

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance

        Parameters:
        - X_test: Test features
        - y_test: True test labels

        Returns:
        - metrics: Dictionary containing evaluation metrics
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, predictions, output_dict=True)
        }

        return metrics

    def get_feature_importance(self):
        """
        Get feature importance scores

        Returns:
        - feature_importance: Array of feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        return self.model.feature_importances_

    def visualize_tree(self, feature_names=None, class_names=None, max_depth=3):
        """
        Visualize the decision tree (requires graphviz)

        Parameters:
        - feature_names: Names of the features
        - class_names: Names of the classes
        - max_depth: Maximum depth to visualize
        """
        from sklearn.tree import plot_tree
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10,
            max_depth=max_depth
        )
        plt.title("Decision Tree Visualization")
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load your dataset here
    # X, y = load_your_data()

    # Split the data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    dt_model = DecisionTreeModel(
        criterion='gini',
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1
    )

    # Train the model
    # dt_model.train(X_train, y_train)

    # Make predictions
    # predictions = dt_model.predict(X_test)

    # Evaluate the model
    # metrics = dt_model.evaluate(X_test, y_test)
    # print(f"Accuracy: {metrics['accuracy']:.4f}")

    # Get feature importance
    # importance = dt_model.get_feature_importance()
    # print("Feature Importance:", importance)

    print("Decision Tree model implementation ready!")
`
  },

  'Logistic Regression': {
    title: 'Logistic Regression Implementation',
    description: 'A comprehensive logistic regression classifier with regularization options',
    code: `# Logistic Regression Classifier Implementation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegressionModel:
    def __init__(self, penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42):
        """
        Initialize Logistic Regression Classifier

        Parameters:
        - penalty: Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
        - C: Inverse of regularization strength (smaller values = stronger regularization)
        - solver: Algorithm for optimization ('liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga')
        - max_iter: Maximum number of iterations
        - random_state: Random state for reproducibility
        """
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.is_scaled = False

    def preprocess_data(self, X_train, X_test=None, scale_features=True):
        """
        Preprocess the training data

        Parameters:
        - X_train: Training features
        - X_test: Test features (optional)
        - scale_features: Whether to scale features

        Returns:
        - X_train_processed, X_test_processed (if provided)
        """
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.is_scaled = True

            if X_test is not None:
                X_test_scaled = self.scaler.transform(X_test)
                return X_train_scaled, X_test_scaled

            return X_train_scaled

        if X_test is not None:
            return X_train, X_test

        return X_train

    def train(self, X_train, y_train, scale_features=True):
        """
        Train the Logistic Regression model

        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - scale_features: Whether to scale features
        """
        # Preprocess data
        X_train_processed = self.preprocess_data(X_train, scale_features=scale_features)

        # Train the model
        self.model.fit(X_train_processed, y_train)
        self.is_trained = True

        print(f"Logistic Regression trained with {len(X_train)} samples")
        print(f"Number of features: {X_train_processed.shape[1]}")
        print(f"Number of iterations: {self.model.n_iter_[0] if hasattr(self.model, 'n_iter_') else 'N/A'}")
        print(f"Convergence: {'Yes' if self.model.n_iter_[0] < self.model.max_iter else 'No'}")

    def predict(self, X_test):
        """
        Make predictions using the trained model

        Parameters:
        - X_test: Test features

        Returns:
        - predictions: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Scale test data if scaler was used during training
        if self.is_scaled:
            X_test_processed = self.scaler.transform(X_test)
        else:
            X_test_processed = X_test

        predictions = self.model.predict(X_test_processed)
        return predictions

    def predict_proba(self, X_test):
        """
        Get prediction probabilities

        Parameters:
        - X_test: Test features

        Returns:
        - probabilities: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Scale test data if scaler was used during training
        if self.is_scaled:
            X_test_processed = self.scaler.transform(X_test)
        else:
            X_test_processed = X_test

        probabilities = self.model.predict_proba(X_test_processed)
        return probabilities

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance

        Parameters:
        - X_test: Test features
        - y_test: True test labels

        Returns:
        - metrics: Dictionary containing evaluation metrics
        """
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')

        # ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, probabilities[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, probabilities, average='weighted', multi_class='ovr')

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }

        return metrics

    def get_coefficients(self, feature_names=None):
        """
        Get model coefficients (weights)

        Parameters:
        - feature_names: Names of the features

        Returns:
        - coefficients: Dictionary or array of coefficients
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting coefficients")

        coef = self.model.coef_[0] if self.model.coef_.shape[0] == 1 else self.model.coef_

        if feature_names is not None:
            return dict(zip(feature_names, coef))

        return coef

    def plot_coefficients(self, feature_names=None, top_n=20):
        """
        Plot feature coefficients

        Parameters:
        - feature_names: Names of the features
        - top_n: Number of top coefficients to plot
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting coefficients")

        coef = self.model.coef_[0] if self.model.coef_.shape[0] == 1 else self.model.coef_[0]

        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(coef))]

        # Get top coefficients by absolute value
        indices = np.argsort(np.abs(coef))[-top_n:]
        top_coef = coef[indices]
        top_features = [feature_names[i] for i in indices]

        plt.figure(figsize=(10, 8))
        colors = ['red' if x < 0 else 'blue' for x in top_coef]
        plt.barh(range(len(top_coef)), top_coef, color=colors)
        plt.yticks(range(len(top_coef)), top_features)
        plt.xlabel('Coefficient Value')
        plt.title(f'Top {top_n} Feature Coefficients')
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load your dataset here
    # X, y = load_your_data()

    # Split the data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    lr_model = LogisticRegressionModel(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000
    )

    # Train the model
    # lr_model.train(X_train, y_train, scale_features=True)

    # Make predictions
    # predictions = lr_model.predict(X_test)

    # Evaluate the model
    # metrics = lr_model.evaluate(X_test, y_test)
    # print(f"Accuracy: {metrics['accuracy']:.4f}")
    # print(f"F1 Score: {metrics['f1_score']:.4f}")

    # Get coefficients
    # coefficients = lr_model.get_coefficients()
    # print("Model Coefficients:", coefficients)

    print("Logistic Regression model implementation ready!")
`
  },

  'SVM': {
    title: 'Support Vector Machine Implementation',
    description: 'A powerful SVM classifier with different kernel options',
    code: `# Support Vector Machine (SVM) Classifier Implementation
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SVMModel:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', degree=3, probability=True, random_state=42):
        """
        Initialize SVM Classifier

        Parameters:
        - C: Regularization parameter (larger C = less regularization)
        - kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        - gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        - degree: Degree of the polynomial kernel function
        - probability: Whether to enable probability estimates
        - random_state: Random state for reproducibility
        """
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            probability=probability,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.is_scaled = False
        self.kernel = kernel

    def preprocess_data(self, X_train, X_test=None, scale_features=True):
        """
        Preprocess the training data (SVM benefits greatly from feature scaling)

        Parameters:
        - X_train: Training features
        - X_test: Test features (optional)
        - scale_features: Whether to scale features (highly recommended for SVM)

        Returns:
        - X_train_processed, X_test_processed (if provided)
        """
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.is_scaled = True

            if X_test is not None:
                X_test_scaled = self.scaler.transform(X_test)
                return X_train_scaled, X_test_scaled

            return X_train_scaled

        if X_test is not None:
            return X_train, X_test

        return X_train

    def train(self, X_train, y_train, scale_features=True):
        """
        Train the SVM model

        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - scale_features: Whether to scale features (recommended for SVM)
        """
        # Preprocess data
        X_train_processed = self.preprocess_data(X_train, scale_features=scale_features)

        # Train the model
        self.model.fit(X_train_processed, y_train)
        self.is_trained = True

        print(f"SVM trained with {len(X_train)} samples")
        print(f"Number of features: {X_train_processed.shape[1]}")
        print(f"Kernel: {self.model.kernel}")
        print(f"Number of support vectors: {self.model.n_support_}")
        print(f"Total support vectors: {len(self.model.support_)}")
        print(f"Support vector ratio: {len(self.model.support_) / len(X_train):.3f}")

    def predict(self, X_test):
        """
        Make predictions using the trained model

        Parameters:
        - X_test: Test features

        Returns:
        - predictions: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Scale test data if scaler was used during training
        if self.is_scaled:
            X_test_processed = self.scaler.transform(X_test)
        else:
            X_test_processed = X_test

        predictions = self.model.predict(X_test_processed)
        return predictions

    def predict_proba(self, X_test):
        """
        Get prediction probabilities (if probability=True)

        Parameters:
        - X_test: Test features

        Returns:
        - probabilities: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if not self.model.probability:
            raise ValueError("Probability estimates are not available. Set probability=True when creating the model.")

        # Scale test data if scaler was used during training
        if self.is_scaled:
            X_test_processed = self.scaler.transform(X_test)
        else:
            X_test_processed = X_test

        probabilities = self.model.predict_proba(X_test_processed)
        return probabilities

    def decision_function(self, X_test):
        """
        Get distance of samples to the separating hyperplane

        Parameters:
        - X_test: Test features

        Returns:
        - decision_scores: Decision function scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before computing decision function")

        # Scale test data if scaler was used during training
        if self.is_scaled:
            X_test_processed = self.scaler.transform(X_test)
        else:
            X_test_processed = X_test

        decision_scores = self.model.decision_function(X_test_processed)
        return decision_scores

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance

        Parameters:
        - X_test: Test features
        - y_test: True test labels

        Returns:
        - metrics: Dictionary containing evaluation metrics
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, predictions)
        }

        return metrics

    def get_support_vectors(self):
        """
        Get support vectors

        Returns:
        - support_vectors: The support vectors
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting support vectors")

        return self.model.support_vectors_

    def hyperparameter_tuning(self, X_train, y_train, param_grid=None, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV

        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - param_grid: Parameter grid for search
        - cv: Number of cross-validation folds

        Returns:
        - best_params: Best parameters found
        """
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }

        # Preprocess data
        X_train_processed = self.preprocess_data(X_train, scale_features=True)

        # Create a new SVM for grid search
        svm_for_search = SVC(probability=True, random_state=self.model.random_state)

        # Perform grid search
        grid_search = GridSearchCV(
            svm_for_search,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train_processed, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True

        return grid_search.best_params_

    def plot_decision_boundary(self, X, y, feature_indices=[0, 1], resolution=0.02):
        """
        Plot decision boundary for 2D data

        Parameters:
        - X: Features
        - y: Labels
        - feature_indices: Which two features to plot
        - resolution: Resolution of the decision boundary plot
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting decision boundary")

        if len(feature_indices) != 2:
            raise ValueError("Exactly two features must be selected for 2D plotting")

        # Select two features
        X_plot = X[:, feature_indices]

        # Scale if necessary
        if self.is_scaled:
            X_plot = self.scaler.transform(X_plot)

        # Create a mesh
        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                            np.arange(y_min, y_max, resolution))

        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        # Extend to full feature space with zeros for other features
        if X.shape[1] > 2:
            mesh_extended = np.zeros((mesh_points.shape[0], X.shape[1]))
            mesh_extended[:, feature_indices] = mesh_points
            Z = self.model.predict(mesh_extended)
        else:
            Z = self.model.predict(mesh_points)

        Z = Z.reshape(xx.shape)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
        scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='black')
        plt.colorbar(scatter)
        plt.xlabel(f'Feature {feature_indices[0]}')
        plt.ylabel(f'Feature {feature_indices[1]}')
        plt.title(f'SVM Decision Boundary ({self.kernel} kernel)')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load your dataset here
    # X, y = load_your_data()

    # Split the data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    svm_model = SVMModel(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True
    )

    # Train the model
    # svm_model.train(X_train, y_train, scale_features=True)

    # Make predictions
    # predictions = svm_model.predict(X_test)

    # Evaluate the model
    # metrics = svm_model.evaluate(X_test, y_test)
    # print(f"Accuracy: {metrics['accuracy']:.4f}")

    # Hyperparameter tuning (optional)
    # best_params = svm_model.hyperparameter_tuning(X_train, y_train)

    print("SVM model implementation ready!")
`
  },

  'K-Nearest Neighbor': {
    title: 'K-Nearest Neighbors Implementation',
    description: 'A comprehensive KNN classifier with distance metrics and optimization',
    code: `# K-Nearest Neighbors (KNN) Classifier Implementation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

class KNNModel:
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto',
                 metric='minkowski', p=2, leaf_size=30):
        """
        Initialize KNN Classifier

        Parameters:
        - n_neighbors: Number of neighbors to use
        - weights: Weight function ('uniform', 'distance')
        - algorithm: Algorithm to compute neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
        - metric: Distance metric ('euclidean', 'manhattan', 'minkowski', 'chebyshev')
        - p: Parameter for Minkowski metric (p=1: Manhattan, p=2: Euclidean)
        - leaf_size: Leaf size for tree algorithms
        """
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            metric=metric,
            p=p,
            leaf_size=leaf_size
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.is_scaled = False
        self.n_neighbors = n_neighbors

    def preprocess_data(self, X_train, X_test=None, scale_features=True):
        """
        Preprocess the training data (KNN is sensitive to feature scaling)

        Parameters:
        - X_train: Training features
        - X_test: Test features (optional)
        - scale_features: Whether to scale features (recommended for KNN)

        Returns:
        - X_train_processed, X_test_processed (if provided)
        """
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.is_scaled = True

            if X_test is not None:
                X_test_scaled = self.scaler.transform(X_test)
                return X_train_scaled, X_test_scaled

            return X_train_scaled

        if X_test is not None:
            return X_train, X_test

        return X_train

    def train(self, X_train, y_train, scale_features=True):
        """
        Train the KNN model (actually just stores the training data)

        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - scale_features: Whether to scale features
        """
        # Preprocess data
        X_train_processed = self.preprocess_data(X_train, scale_features=scale_features)

        # "Train" the model (KNN is a lazy learner - just stores the data)
        self.model.fit(X_train_processed, y_train)
        self.is_trained = True

        print(f"KNN trained with {len(X_train)} samples")
        print(f"Number of features: {X_train_processed.shape[1]}")
        print(f"Number of neighbors (k): {self.model.n_neighbors}")
        print(f"Distance metric: {self.model.metric}")
        print(f"Weight function: {self.model.weights}")
        print(f"Algorithm: {self.model.algorithm}")

        # Analyze class distribution
        class_counts = Counter(y_train)
        print(f"Class distribution: {dict(class_counts)}")

    def predict(self, X_test):
        """
        Make predictions using the trained model

        Parameters:
        - X_test: Test features

        Returns:
        - predictions: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Scale test data if scaler was used during training
        if self.is_scaled:
            X_test_processed = self.scaler.transform(X_test)
        else:
            X_test_processed = X_test

        predictions = self.model.predict(X_test_processed)
        return predictions

    def predict_proba(self, X_test):
        """
        Get prediction probabilities

        Parameters:
        - X_test: Test features

        Returns:
        - probabilities: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Scale test data if scaler was used during training
        if self.is_scaled:
            X_test_processed = self.scaler.transform(X_test)
        else:
            X_test_processed = X_test

        probabilities = self.model.predict_proba(X_test_processed)
        return probabilities

    def get_neighbors(self, X_test, return_distance=True):
        """
        Get k nearest neighbors for test samples

        Parameters:
        - X_test: Test features
        - return_distance: Whether to return distances

        Returns:
        - distances, indices: Distances and indices of neighbors
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting neighbors")

        # Scale test data if scaler was used during training
        if self.is_scaled:
            X_test_processed = self.scaler.transform(X_test)
        else:
            X_test_processed = X_test

        if return_distance:
            distances, indices = self.model.kneighbors(X_test_processed, return_distance=True)
            return distances, indices
        else:
            indices = self.model.kneighbors(X_test_processed, return_distance=False)
            return indices

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance

        Parameters:
        - X_test: Test features
        - y_test: True test labels

        Returns:
        - metrics: Dictionary containing evaluation metrics
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, predictions)
        }

        return metrics

    def find_optimal_k(self, X_train, y_train, k_range=range(1, 31), cv=5, scale_features=True):
        """
        Find the optimal number of neighbors using cross-validation

        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - k_range: Range of k values to test
        - cv: Number of cross-validation folds
        - scale_features: Whether to scale features

        Returns:
        - best_k: Optimal number of neighbors
        - scores: Cross-validation scores for each k
        """
        # Preprocess data
        X_train_processed = self.preprocess_data(X_train, scale_features=scale_features)

        scores = []
        for k in k_range:
            knn = KNeighborsClassifier(
                n_neighbors=k,
                weights=self.model.weights,
                algorithm=self.model.algorithm,
                metric=self.model.metric,
                p=self.model.p
            )
            cv_scores = cross_val_score(knn, X_train_processed, y_train, cv=cv)
            scores.append(cv_scores.mean())

        best_k = k_range[np.argmax(scores)]

        print(f"Optimal k: {best_k}")
        print(f"Best CV score: {max(scores):.4f}")

        # Update model with optimal k
        self.model.set_params(n_neighbors=best_k)
        self.n_neighbors = best_k

        return best_k, scores

    def plot_k_optimization(self, k_range, scores):
        """
        Plot the cross-validation scores for different k values

        Parameters:
        - k_range: Range of k values tested
        - scores: Cross-validation scores
        """
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, scores, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('KNN: Accuracy vs Number of Neighbors')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_range[::2])  # Show every second k value

        # Highlight the best k
        best_k = k_range[np.argmax(scores)]
        best_score = max(scores)
        plt.scatter(best_k, best_score, color='red', s=100, zorder=5)
        plt.annotate(f'Best k={best_k}\\nScore={best_score:.3f}',
                    xy=(best_k, best_score),
                    xytext=(best_k + 3, best_score - 0.01),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        plt.tight_layout()
        plt.show()

    def analyze_prediction(self, X_test_sample, y_true=None):
        """
        Analyze a single prediction by showing the nearest neighbors

        Parameters:
        - X_test_sample: Single test sample (should be 2D array)
        - y_true: True label (optional)

        Returns:
        - analysis: Dictionary with prediction analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing predictions")

        # Make prediction
        prediction = self.predict(X_test_sample)[0]
        probabilities = self.predict_proba(X_test_sample)[0]

        # Get neighbors
        distances, indices = self.get_neighbors(X_test_sample, return_distance=True)
        neighbor_distances = distances[0]
        neighbor_indices = indices[0]

        # Get neighbor labels
        neighbor_labels = [self.model._y[idx] for idx in neighbor_indices]

        analysis = {
            'prediction': prediction,
            'probabilities': dict(zip(self.model.classes_, probabilities)),
            'neighbor_distances': neighbor_distances,
            'neighbor_labels': neighbor_labels,
            'neighbor_indices': neighbor_indices
        }

        if y_true is not None:
            analysis['true_label'] = y_true
            analysis['correct'] = prediction == y_true

        # Print analysis
        print(f"Prediction: {prediction}")
        if y_true is not None:
            print(f"True label: {y_true} ({'Correct' if prediction == y_true else 'Incorrect'})")

        print(f"Prediction probabilities:")
        for class_label, prob in analysis['probabilities'].items():
            print(f"  Class {class_label}: {prob:.3f}")

        print(f"\\nNearest {self.n_neighbors} neighbors:")
        for i, (dist, label) in enumerate(zip(neighbor_distances, neighbor_labels)):
            print(f"  Neighbor {i+1}: Label {label}, Distance {dist:.3f}")

        return analysis

    def plot_decision_boundary(self, X, y, feature_indices=[0, 1], resolution=0.02):
        """
        Plot decision boundary for 2D data

        Parameters:
        - X: Features
        - y: Labels
        - feature_indices: Which two features to plot
        - resolution: Resolution of the decision boundary plot
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting decision boundary")

        if len(feature_indices) != 2:
            raise ValueError("Exactly two features must be selected for 2D plotting")

        # Select two features
        X_plot = X[:, feature_indices]

        # Scale if necessary
        if self.is_scaled:
            X_plot = self.scaler.transform(X_plot)

        # Create a mesh
        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                            np.arange(y_min, y_max, resolution))

        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        # Extend to full feature space with zeros for other features
        if X.shape[1] > 2:
            mesh_extended = np.zeros((mesh_points.shape[0], X.shape[1]))
            mesh_extended[:, feature_indices] = mesh_points
            Z = self.model.predict(mesh_extended)
        else:
            Z = self.model.predict(mesh_points)

        Z = Z.reshape(xx.shape)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
        scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='black')
        plt.colorbar(scatter)
        plt.xlabel(f'Feature {feature_indices[0]}')
        plt.ylabel(f'Feature {feature_indices[1]}')
        plt.title(f'KNN Decision Boundary (k={self.n_neighbors})')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load your dataset here
    # X, y = load_your_data()

    # Split the data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    knn_model = KNNModel(
        n_neighbors=5,
        weights='uniform',
        algorithm='auto',
        metric='minkowski'
    )

    # Find optimal k
    # best_k, scores = knn_model.find_optimal_k(X_train, y_train, k_range=range(1, 21))
    # knn_model.plot_k_optimization(range(1, 21), scores)

    # Train the model
    # knn_model.train(X_train, y_train, scale_features=True)

    # Make predictions
    # predictions = knn_model.predict(X_test)

    # Evaluate the model
    # metrics = knn_model.evaluate(X_test, y_test)
    # print(f"Accuracy: {metrics['accuracy']:.4f}")

    # Analyze a single prediction
    # analysis = knn_model.analyze_prediction(X_test[[0]], y_test[0])

    print("KNN model implementation ready!")
`
  },

  'Artificial Neural Network': {
    title: 'Artificial Neural Network Implementation',
    description: 'A multi-layer perceptron classifier with backpropagation and various activation functions',
    code: `# Artificial Neural Network (Multi-Layer Perceptron) Implementation
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ANNModel:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam',
                 alpha=0.0001, learning_rate='constant', learning_rate_init=0.001,
                 max_iter=200, early_stopping=True, validation_fraction=0.1,
                 n_iter_no_change=10, random_state=42):
        """
        Initialize Artificial Neural Network (Multi-Layer Perceptron) Classifier

        Parameters:
        - hidden_layer_sizes: Tuple of hidden layer sizes (e.g., (100,) for one layer with 100 neurons)
        - activation: Activation function ('relu', 'tanh', 'logistic', 'identity')
        - solver: Solver for weight optimization ('adam', 'sgd', 'lbfgs')
        - alpha: L2 penalty (regularization) parameter
        - learning_rate: Learning rate schedule ('constant', 'invscaling', 'adaptive')
        - learning_rate_init: Initial learning rate
        - max_iter: Maximum number of iterations
        - early_stopping: Whether to use early stopping
        - validation_fraction: Fraction of training data for early stopping
        - n_iter_no_change: Number of iterations with no improvement to wait
        - random_state: Random state for reproducibility
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.is_scaled = False
        self.training_history = []

    def preprocess_data(self, X_train, X_test=None, scale_features=True):
        """
        Preprocess the training data (Neural networks require feature scaling)

        Parameters:
        - X_train: Training features
        - X_test: Test features (optional)
        - scale_features: Whether to scale features (required for ANN)

        Returns:
        - X_train_processed, X_test_processed (if provided)
        """
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.is_scaled = True

            if X_test is not None:
                X_test_scaled = self.scaler.transform(X_test)
                return X_train_scaled, X_test_scaled

            return X_train_scaled

        if X_test is not None:
            return X_train, X_test

        return X_train

    def train(self, X_train, y_train, scale_features=True, verbose=True):
        """
        Train the ANN model

        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - scale_features: Whether to scale features
        - verbose: Whether to print training information
        """
        # Preprocess data
        X_train_processed = self.preprocess_data(X_train, scale_features=scale_features)

        # Encode labels if they are strings
        if y_train.dtype == 'object' or isinstance(y_train[0], str):
            y_train_encoded = self.label_encoder.fit_transform(y_train)
        else:
            y_train_encoded = y_train

        # Train the model
        self.model.fit(X_train_processed, y_train_encoded)
        self.is_trained = True

        if verbose:
            print(f"ANN trained with {len(X_train)} samples")
            print(f"Number of features: {X_train_processed.shape[1]}")
            print(f"Hidden layer sizes: {self.model.hidden_layer_sizes}")
            print(f"Activation function: {self.model.activation}")
            print(f"Solver: {self.model.solver}")
            print(f"Number of iterations: {self.model.n_iter_}")
            print(f"Final loss: {self.model.loss_:.6f}")

            if hasattr(self.model, 'validation_scores_'):
                print(f"Best validation score: {max(self.model.validation_scores_):.4f}")

            # Analyze network architecture
            total_params = 0
            layer_info = []

            # Input to first hidden layer
            if len(self.model.coefs_) > 0:
                input_size = self.model.coefs_[0].shape[0]
                first_hidden = self.model.coefs_[0].shape[1]
                params = input_size * first_hidden + first_hidden  # weights + biases
                total_params += params
                layer_info.append(f"Input ({input_size}) -> Hidden1 ({first_hidden}): {params} params")

                # Hidden layers
                for i in range(1, len(self.model.coefs_)):
                    prev_size = self.model.coefs_[i-1].shape[1]
                    curr_size = self.model.coefs_[i].shape[1]
                    params = prev_size * curr_size + curr_size
                    total_params += params
                    layer_info.append(f"Hidden{i} ({prev_size}) -> Hidden{i+1} ({curr_size}): {params} params")

            print(f"\\nNetwork Architecture:")
            for info in layer_info:
                print(f"  {info}")
            print(f"Total parameters: {total_params}")

    def predict(self, X_test):
        """
        Make predictions using the trained model

        Parameters:
        - X_test: Test features

        Returns:
        - predictions: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Scale test data if scaler was used during training
        if self.is_scaled:
            X_test_processed = self.scaler.transform(X_test)
        else:
            X_test_processed = X_test

        predictions_encoded = self.model.predict(X_test_processed)

        # Decode predictions if label encoder was used
        if hasattr(self.label_encoder, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions_encoded)
        else:
            predictions = predictions_encoded

        return predictions

    def predict_proba(self, X_test):
        """
        Get prediction probabilities

        Parameters:
        - X_test: Test features

        Returns:
        - probabilities: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Scale test data if scaler was used during training
        if self.is_scaled:
            X_test_processed = self.scaler.transform(X_test)
        else:
            X_test_processed = X_test

        probabilities = self.model.predict_proba(X_test_processed)
        return probabilities

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance

        Parameters:
        - X_test: Test features
        - y_test: True test labels

        Returns:
        - metrics: Dictionary containing evaluation metrics
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'training_loss': self.model.loss_ if hasattr(self.model, 'loss_') else None,
            'n_iterations': self.model.n_iter_ if hasattr(self.model, 'n_iter_') else None
        }

        return metrics

    def plot_training_history(self):
        """
        Plot training and validation loss curves (if available)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting training history")

        plt.figure(figsize=(12, 4))

        # Plot loss curve
        plt.subplot(1, 2, 1)
        if hasattr(self.model, 'loss_curve_'):
            plt.plot(self.model.loss_curve_, label='Training Loss', linewidth=2)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Loss curve not available', ha='center', va='center')
            plt.title('Training Loss Curve')

        # Plot validation score curve
        plt.subplot(1, 2, 2)
        if hasattr(self.model, 'validation_scores_'):
            plt.plot(self.model.validation_scores_, label='Validation Score', linewidth=2, color='orange')
            plt.xlabel('Iterations')
            plt.ylabel('Validation Score')
            plt.title('Validation Score Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Validation scores not available', ha='center', va='center')
            plt.title('Validation Score Curve')

        plt.tight_layout()
        plt.show()

    def get_layer_weights(self, layer_index=0):
        """
        Get weights of a specific layer

        Parameters:
        - layer_index: Index of the layer (0 for first hidden layer)

        Returns:
        - weights: Weight matrix of the specified layer
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting weights")

        if layer_index >= len(self.model.coefs_):
            raise ValueError(f"Layer index {layer_index} is out of range. Model has {len(self.model.coefs_)} layers.")

        return self.model.coefs_[layer_index]

    def get_layer_biases(self, layer_index=0):
        """
        Get biases of a specific layer

        Parameters:
        - layer_index: Index of the layer (0 for first hidden layer)

        Returns:
        - biases: Bias vector of the specified layer
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting biases")

        if layer_index >= len(self.model.intercepts_):
            raise ValueError(f"Layer index {layer_index} is out of range. Model has {len(self.model.intercepts_)} layers.")

        return self.model.intercepts_[layer_index]

    def visualize_network_weights(self, layer_index=0, max_neurons=20):
        """
        Visualize the weights of a specific layer

        Parameters:
        - layer_index: Index of the layer to visualize
        - max_neurons: Maximum number of neurons to visualize
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before visualizing weights")

        weights = self.get_layer_weights(layer_index)

        # Limit the number of neurons to visualize
        n_neurons = min(weights.shape[1], max_neurons)
        weights_subset = weights[:, :n_neurons]

        plt.figure(figsize=(12, 8))
        plt.imshow(weights_subset.T, cmap='RdBu', aspect='auto')
        plt.colorbar(label='Weight Value')
        plt.xlabel('Input Features')
        plt.ylabel(f'Neurons in Layer {layer_index + 1}')
        plt.title(f'Weight Visualization for Layer {layer_index + 1}')
        plt.tight_layout()
        plt.show()

    def hyperparameter_tuning(self, X_train, y_train, param_name='alpha',
                            param_range=np.logspace(-4, -1, 10), cv=5):
        """
        Perform hyperparameter tuning using validation curves

        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - param_name: Parameter to tune
        - param_range: Range of parameter values to test
        - cv: Number of cross-validation folds

        Returns:
        - best_param: Best parameter value
        - train_scores: Training scores for each parameter
        - val_scores: Validation scores for each parameter
        """
        # Preprocess data
        X_train_processed = self.preprocess_data(X_train, scale_features=True)

        # Encode labels if necessary
        if y_train.dtype == 'object' or isinstance(y_train[0], str):
            y_train_encoded = self.label_encoder.fit_transform(y_train)
        else:
            y_train_encoded = y_train

        # Create a new model for tuning
        model_for_tuning = MLPClassifier(
            hidden_layer_sizes=self.model.hidden_layer_sizes,
            activation=self.model.activation,
            solver=self.model.solver,
            max_iter=100,  # Reduce for faster tuning
            random_state=self.model.random_state
        )

        # Perform validation curve analysis
        train_scores, val_scores = validation_curve(
            model_for_tuning, X_train_processed, y_train_encoded,
            param_name=param_name, param_range=param_range,
            cv=cv, scoring='accuracy', n_jobs=-1
        )

        # Calculate mean scores
        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)

        # Find best parameter
        best_idx = np.argmax(val_mean)
        best_param = param_range[best_idx]

        print(f"Best {param_name}: {best_param}")
        print(f"Best validation score: {val_mean[best_idx]:.4f}")

        # Plot validation curve
        plt.figure(figsize=(10, 6))
        plt.semilogx(param_range, train_mean, 'bo-', label='Training Score')
        plt.semilogx(param_range, val_mean, 'ro-', label='Validation Score')
        plt.fill_between(param_range, train_mean - np.std(train_scores, axis=1),
                        train_mean + np.std(train_scores, axis=1), alpha=0.1, color='blue')
        plt.fill_between(param_range, val_mean - np.std(val_scores, axis=1),
                        val_mean + np.std(val_scores, axis=1), alpha=0.1, color='red')
        plt.xlabel(param_name)
        plt.ylabel('Accuracy')
        plt.title(f'Validation Curve for {param_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return best_param, train_scores, val_scores

    def predict_single_sample(self, sample, show_probabilities=True):
        """
        Make prediction for a single sample with detailed output

        Parameters:
        - sample: Single sample to predict (1D array)
        - show_probabilities: Whether to show class probabilities

        Returns:
        - prediction_info: Dictionary with prediction details
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Reshape sample to 2D if needed
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        prediction = self.predict(sample)[0]

        prediction_info = {
            'prediction': prediction,
            'input_shape': sample.shape
        }

        if show_probabilities:
            probabilities = self.predict_proba(sample)[0]
            classes = self.model.classes_
            if hasattr(self.label_encoder, 'classes_'):
                classes = self.label_encoder.classes_

            prediction_info['probabilities'] = dict(zip(classes, probabilities))
            prediction_info['confidence'] = max(probabilities)

        print(f"Prediction: {prediction}")
        if show_probabilities:
            print("Class probabilities:")
            for class_name, prob in prediction_info['probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
            print(f"Confidence: {prediction_info['confidence']:.4f}")

        return prediction_info

# Example usage
if __name__ == "__main__":
    # Load your dataset here
    # X, y = load_your_data()

    # Split the data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    ann_model = ANNModel(
        hidden_layer_sizes=(100, 50),  # Two hidden layers
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=200,
        early_stopping=True
    )

    # Train the model
    # ann_model.train(X_train, y_train, scale_features=True, verbose=True)

    # Plot training history
    # ann_model.plot_training_history()

    # Make predictions
    # predictions = ann_model.predict(X_test)

    # Evaluate the model
    # metrics = ann_model.evaluate(X_test, y_test)
    # print(f"Accuracy: {metrics['accuracy']:.4f}")

    # Hyperparameter tuning (optional)
    # best_alpha, train_scores, val_scores = ann_model.hyperparameter_tuning(
    #     X_train, y_train, param_name='alpha', param_range=np.logspace(-4, -1, 10)
    # )

    # Visualize network weights
    # ann_model.visualize_network_weights(layer_index=0)

    print("ANN model implementation ready!")
`
  }
};

const CodeViewer = ({ isOpen, onClose, modelName, isDarkMode = false }) => {
  // Model name is received as prop, no need for state

  if (!isOpen) return null;

  const modelData = MODEL_SOURCE_CODES[modelName];

  if (!modelData) {
    return (
      <div className="code-viewer-overlay">
        <div className="code-viewer-modal">
          <div className="code-viewer-header">
            <h3>Model Source Code</h3>
            <button className="close-btn" onClick={onClose}>&times;</button>
          </div>
          <div className="code-viewer-content">
            <p>Source code not available for {modelName}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="code-viewer-overlay">
      <div className="code-viewer-modal">
        <div className="code-viewer-header">
          <div className="header-info">
            <h3>{modelData.title}</h3>
            <p className="model-description">{modelData.description}</p>
          </div>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>

        <div className="code-viewer-content">
          <div className="code-actions">
            <button
              className="copy-btn"
              onClick={() => {
                navigator.clipboard.writeText(modelData.code);
                alert('Code copied to clipboard!');
              }}
            >
              📋 Copy Code
            </button>
            <span className="language-tag">Python</span>
          </div>

          <div className="code-container">
            <SyntaxHighlighter
              language="python"
              style={isDarkMode ? vscDarkPlus : vs}
              showLineNumbers={true}
              wrapLines={true}
              customStyle={{
                margin: 0,
                borderRadius: '8px',
                fontSize: '13px',
                lineHeight: '1.4'
              }}
            >
              {modelData.code}
            </SyntaxHighlighter>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CodeViewer;
