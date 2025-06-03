# In linear_regression/model_evaluator.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Make sure this function is defined in this file or imported correctly
# if it's used by evaluate_model.
# For example:
def calculate_sst_ssr_sse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mean = np.mean(y_true)
    sst = np.sum((y_true - y_mean)**2)
    sse = np.sum((y_true - y_pred)**2)
    ssr = sst - sse # A common definition where SST = SSR + SSE
    return sst, ssr, sse

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Calculates, prints, and returns common regression evaluation metrics.

    Args:
        y_true (pd.Series or np.array): Actual target values.
        y_pred (np.array): Predicted target values.
        model_name (str): Name of the model being evaluated (used for printing).

    Returns:
        dict: A dictionary containing the calculated metrics:
              {'MSE', 'MAE', 'R2 Score', 'RMSE'}.
              It can also include 'SST', 'SSR', 'SSE' if you choose to add them.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)  # Added RMSE

    # This dictionary will be returned for aggregation
    metrics_to_return = {
        "MSE": mse,
        "MAE": mae,
        "R2 Score": r2,
        "RMSE": rmse
    }

    # Printing can remain as is, providing immediate feedback
    print(f"\n--- Evaluation Metrics for {model_name} ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}") # Added RMSE here too
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    # For Linear Regression, you can also include the SST, SSR, SSE calculation
    # This part handles the specific printing for Linear Regression.
    # If you want SST, SSR, SSE in your final summary table,
    # you should add them to the 'metrics_to_return' dictionary.
    if "Linear Regression" in model_name and "Test" not in model_name:
        try:
            # Assuming calculate_sst_ssr_sse is defined and accessible
            sst, ssr, sse = calculate_sst_ssr_sse(y_true, y_pred)
            print(f"Total Sum of Squares (SST): {sst:.2f}")
            print(f"Regression Sum of Squares (SSR): {ssr:.2f}")
            print(f"Error Sum of Squares (SSE): {sse:.2f}")
            # Optional: Add to returned dictionary if you want these in the final summary table
            # metrics_to_return["SST"] = sst
            # metrics_to_return["SSR"] = ssr
            # metrics_to_return["SSE"] = sse
        except NameError:
            print("Warning: 'calculate_sst_ssr_sse' function not found. Skipping SST/SSR/SSE calculation.")
        except Exception as e:
            print(f"Warning: Error calculating SST/SSR/SSE: {e}")
    
    print("-" * (28 + len(model_name))) # Prints a closing line for the metrics block

    return metrics_to_return