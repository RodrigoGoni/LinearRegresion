import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def plot_predictions_vs_actual(y_true, y_pred, title, save_path):
    """
    Generates and saves a scatter plot of predictions vs. actual values.

    Args:
        y_true (pd.Series or np.array): Actual target values.
        y_pred (np.array): Predicted target values.
        title (str): Title of the plot.
        save_path (str): Full path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha': 0.3})
    plt.xlabel("Valor Real de MedHouseVal")
    plt.ylabel("Predicción de MedHouseVal")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def plot_residuals_histogram(y_true, y_pred, title, save_path):
    """
    Generates and saves a histogram of residuals.

    Args:
        y_true (pd.Series or np.array): Actual target values.
        y_pred (np.array): Predicted target values.
        title (str): Title of the plot.
        save_path (str): Full path to save the plot.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuos")
    plt.ylabel("Frecuencia")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def plot_residuals_vs_predicted(y_pred, residuals, title, save_path):
    """
    Generates and saves a scatter plot of residuals vs. predicted values.

    Args:
        y_pred (np.array): Predicted target values.
        residuals (np.array): Residuals (actual - predicted).
        title (str): Title of the plot.
        save_path (str): Full path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Valores Predichos")
    plt.ylabel("Residuos")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()



def plot_ridge_mse_vs_alpha(cv_results, best_alpha, output_plot_dir, filename_prefix=""): # NEW SIGNATURE
    """
    Plots Mean Squared Error (MSE) vs. alpha values from Ridge/Lasso cross-validation.

    Args:
        cv_results (dict): Results from GridSearchCV. Expects 'param_alpha' and 'mean_test_score'.
        best_alpha (float): The best alpha value determined.
        output_plot_dir (str): Directory to save the plot.
        filename_prefix (str, optional): Prefix for the plot filename. Defaults to "".
    """
    try:
        # Ensure 'param_alpha' exists and extract alphas
        if 'param_alpha' in cv_results:
            # Convert to float, handling potential MaskedArray from GridSearchCV
            if isinstance(cv_results['param_alpha'], np.ma.MaskedArray):
                alphas = cv_results['param_alpha'].data.astype(float)
            else:
                alphas = np.array(cv_results['param_alpha'], dtype=float)
        else:
            print(f"Warning: 'param_alpha' not found in cv_results for {filename_prefix} plot.")
            return

        # Ensure 'mean_test_score' exists and extract scores (typically negative MSE)
        if 'mean_test_score' in cv_results:
            # Scores from GridSearchCV are often negative (e.g., neg_mean_squared_error)
            mse_scores = -np.array(cv_results['mean_test_score'], dtype=float)
            y_axis_label = 'Mean Squared Error (MSE)'
        else:
            print(f"Warning: 'mean_test_score' not found in cv_results for {filename_prefix} plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(alphas, mse_scores, marker='o', linestyle='-', label='MSE per alpha')
        
        # Highlight the best alpha
        plt.axvline(best_alpha, color='r', linestyle='--', 
                    label=f'Best alpha = {best_alpha:.4e}\nMSE = {mse_scores[alphas == best_alpha].min():.4f}')

        plt.xlabel('Alpha (Regularization Strength)')
        plt.ylabel(y_axis_label)
        
        title_core = "MSE vs. Alpha"
        if filename_prefix:
            # Convert prefix like "Ridge_Some_Data" to "Ridge Some Data" for title
            descriptive_name = filename_prefix.replace('_', ' ')
            plt.title(f'{descriptive_name}: {title_core}')
        else:
            plt.title(title_core)
        
        if len(alphas) > 1 and alphas.min() > 0 and alphas.max() / alphas.min() > 100: # Heuristic for log scale
            plt.xscale('log')
            plt.xlabel('Alpha (Regularization Strength) - Log Scale')

        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5) # Grid for both major and minor ticks on log scale
        plt.tight_layout()

        # Construct filename
        base_plot_filename = "mse_vs_alpha.png"
        if filename_prefix:
            final_plot_filename = f"{filename_prefix}_{base_plot_filename}"
        else:
            final_plot_filename = base_plot_filename
        
        plot_path = os.path.join(output_plot_dir, final_plot_filename)

        # Ensure output directory exists
        os.makedirs(output_plot_dir, exist_ok=True)
        
        plt.savefig(plot_path)
        plt.close()
        print(f"    Alpha vs MSE plot saved: {plot_path}")

    except Exception as e:
        print(f"Error generating MSE vs. Alpha plot for {filename_prefix}: {e}")
        # Optionally, print more details of cv_results keys if debugging
        # print(f"Available keys in cv_results: {list(cv_results.keys())}")
