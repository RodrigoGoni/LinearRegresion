from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def calculate_sst_ssr_sse(y_true, y_pred):
    """
    Calculates Total Sum of Squares (SST), Explained Sum of Squares (SSR),
    and Residual Sum of Squares (SSE).
    """
    y_mean = np.mean(y_true)
    sst = np.sum((y_true - y_mean)**2)
    ssr = np.sum((y_pred - y_mean)**2)
    sse = np.sum((y_true - y_pred)**2)
    return sst, ssr, sse


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Calculates and prints common regression evaluation metrics.

    Args:
        y_true (pd.Series or np.array): Actual target values.
        y_pred (np.array): Predicted target values.
        model_name (str): Name of the model being evaluated.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"--- Evaluation Metrics for {model_name} ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    # For Linear Regression, you can also include the SST, SSR, SSE calculation
    if "Linear Regression" in model_name and "Test" not in model_name:  # Only for initial training evaluation
        sst, ssr, sse = calculate_sst_ssr_sse(y_true, y_pred)
        print(f"Total Variance (SST): {sst:.2f}")
        print(f"Explained Variance (SSR): {ssr:.2f}")
        print(f"Residual Variance (SSE): {sse:.2f}")
