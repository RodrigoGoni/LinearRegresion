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


def plot_ridge_mse_vs_alpha(cv_results, best_alpha, output_dir):
    """
    Generates and saves a plot of MSE vs. alpha for Ridge Regression.

    Args:
        cv_results (dict): Results from GridSearchCV's cv_results_.
        best_alpha (float): The best alpha value found.
        output_dir (str): Directory to save the plot.
    """
    alphas = cv_results['param_alpha'].data.astype(float)
    mean_test_mse_cv = -cv_results['mean_test_score']

    plt.figure(figsize=(12, 7))
    plt.plot(alphas, mean_test_mse_cv, marker='o', linestyle='-', markersize=4)
    plt.title(
        'MSE en función de α para Ridge Regression (Validación Cruzada en Entrenamiento)')
    plt.xlabel('Valor de α (Fuerza de Regularización)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xscale('linear')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axvline(best_alpha, color='r', linestyle='--',
                label=f'Mejor α = {best_alpha:.4f}')
    plt.axhline(np.min(mean_test_mse_cv), color='g', linestyle=':',
                label=f'MSE Mínimo CV = {np.min(mean_test_mse_cv):.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ridge_mse_vs_alpha_cv.png'))
    plt.close()
