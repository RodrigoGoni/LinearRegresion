from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np


def train_linear_regression(X_train, y_train, X_test):
    """
    Trains a Linear Regression model and makes predictions.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.

    Returns:
        tuple: A tuple containing:
            - model (LinearRegression): The trained Linear Regression model.
            - y_pred_train (np.array): Predictions on the training set.
            - y_pred_test (np.array): Predictions on the test set.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return model, y_pred_train, y_pred_test


def train_ridge_regression(X_train, y_train):
    """
    Trains a Ridge Regression model using GridSearchCV to find the best alpha.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        tuple: A tuple containing:
            - best_alpha (float): The optimal alpha value found.
            - final_ridge_model (Ridge): The trained Ridge model with the best alpha.
            - cv_results (dict): Results from GridSearchCV.
    """
    alpha_range = np.linspace(0, 12.5, 100)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge = Ridge()

    grid_search = GridSearchCV(
        estimator=ridge,
        param_grid={'alpha': alpha_range},
        cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    print("Starting GridSearchCV for Ridge Regression...")
    grid_search.fit(X_train, y_train)

    best_alpha = grid_search.best_params_['alpha']
    best_cv_mse = -grid_search.best_score_

    print(f"Best alpha found (via CV on training): {best_alpha:.4f}")
    print(f"Lowest MSE from CV (average across folds): {best_cv_mse:.4f}")

    final_ridge_model = Ridge(alpha=best_alpha)
    final_ridge_model.fit(X_train, y_train)

    return best_alpha, final_ridge_model, grid_search.cv_results_

def train_lasso_regression(X_train, y_train):
    """
    Trains a Lasso Regression model using GridSearchCV to find the best alpha.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        tuple: A tuple containing:
            - best_alpha (float): The optimal alpha value found.
            - final_lasso_model (Lasso): The trained Lasso model with the best alpha.
            - cv_results (dict): Results from GridSearchCV.
    """

    alpha_range = np.linspace(0.0001, 1, 100) 
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lasso = Lasso(max_iter=10000)

    grid_search = GridSearchCV(
        estimator=lasso,
        param_grid={'alpha': alpha_range},
        cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0 
    )

    print("Starting GridSearchCV for Lasso Regression...")
    grid_search.fit(X_train, y_train)

    best_alpha = grid_search.best_params_['alpha']
    best_cv_mse = -grid_search.best_score_

    print(f"Best alpha found (via CV on training): {best_alpha:.4f}")
    print(f"Lowest MSE from CV (average across folds): {best_cv_mse:.4f}")

    final_lasso_model = Lasso(alpha=best_alpha, max_iter=10000)
    final_lasso_model.fit(X_train, y_train)

    return best_alpha, final_lasso_model, grid_search.cv_results_