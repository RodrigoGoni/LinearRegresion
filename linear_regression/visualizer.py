import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
def plot_diagnostic_subplots(y_train_true, y_train_pred, y_test_true, y_test_pred, model_name, save_path):
    """
    Generates and saves a single figure with 6 diagnostic plots for a model.

    The figure contains a 2x3 grid:
    - Row 1 (Train Set): Predictions vs. Actual, Residuals Histogram, Residuals vs. Predicted
    - Row 2 (Test Set):  Predictions vs. Actual, Residuals Histogram, Residuals vs. Predicted

    Args:
        y_train_true: Actual training values.
        y_train_pred: Predicted training values.
        y_test_true: Actual testing values.
        y_test_pred: Predicted testing values.
        model_name (str): The full name of the model for the main title.
        save_path (str): Full path to save the consolidated plot.
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(f'Diagnostic Plots for {model_name}', fontsize=20)

    # --- Train Set Calculations ---
    residuals_train = y_train_true - y_train_pred

    # --- Test Set Calculations ---
    residuals_test = y_test_true - y_test_pred

    # --- Row 1: Training Set Plots ---

    # 1.1 Predictions vs. Actual (Train)
    sns.regplot(x=y_train_true, y=y_train_pred, ax=axes[0, 0], scatter_kws={'alpha': 0.3})
    axes[0, 0].set_title('Train: Predictions vs. Actual Values', fontsize=14)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')

    # 1.2 Residuals Histogram (Train)
    sns.histplot(residuals_train, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Train: Residuals Distribution', fontsize=14)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')

    # 1.3 Residuals vs. Predicted (Train)
    sns.scatterplot(x=y_train_pred, y=residuals_train, ax=axes[0, 2], alpha=0.3)
    axes[0, 2].axhline(y=0, color='r', linestyle='--')
    axes[0, 2].set_title('Train: Residuals vs. Predicted Values', fontsize=14)
    axes[0, 2].set_xlabel('Predicted Values')
    axes[0, 2].set_ylabel('Residuals')

    # --- Row 2: Test Set Plots ---

    # 2.1 Predictions vs. Actual (Test)
    sns.regplot(x=y_test_true, y=y_test_pred, ax=axes[1, 0], scatter_kws={'alpha': 0.3})
    axes[1, 0].set_title('Test: Predictions vs. Actual Values', fontsize=14)
    axes[1, 0].set_xlabel('Actual Values')
    axes[1, 0].set_ylabel('Predicted Values')

    # 2.2 Residuals Histogram (Test)
    sns.histplot(residuals_test, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Test: Residuals Distribution', fontsize=14)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')

    # 2.3 Residuals vs. Predicted (Test)
    sns.scatterplot(x=y_test_pred, y=residuals_test, ax=axes[1, 2], alpha=0.3)
    axes[1, 2].axhline(y=0, color='r', linestyle='--')
    axes[1, 2].set_title('Test: Residuals vs. Predicted Values', fontsize=14)
    axes[1, 2].set_xlabel('Predicted Values')
    axes[1, 2].set_ylabel('Residuals')

    # --- Final Touches ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.savefig(save_path)
    plt.close(fig)



def plot_ridge_mse_vs_alpha(cv_results, best_alpha, output_plot_dir, filename_prefix=""):
    """
    Plots Mean Squared Error (MSE) vs. alpha values from Ridge/Lasso cross-validation.

    Args:
        cv_results (dict): Results from GridSearchCV. Expects 'param_alpha' and 'mean_test_score'.
        best_alpha (float): The best alpha value determined.
        output_plot_dir (str): Directory to save the plot.
        filename_prefix (str, optional): Prefix for the plot filename. Defaults to "".
    """
    try:
        if 'param_alpha' in cv_results:
            if isinstance(cv_results['param_alpha'], np.ma.MaskedArray):
                alphas = cv_results['param_alpha'].data.astype(float)
            else:
                alphas = np.array(cv_results['param_alpha'], dtype=float)
        else:
            print(f"Warning: 'param_alpha' not found in cv_results for {filename_prefix} plot.")
            return

        if 'mean_test_score' in cv_results:
            mse_scores = -np.array(cv_results['mean_test_score'], dtype=float)
            y_axis_label = 'Mean Squared Error (MSE)'
        else:
            print(f"Warning: 'mean_test_score' not found in cv_results for {filename_prefix} plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(alphas, mse_scores, marker='o', linestyle='-', label='MSE per alpha')
        
        plt.axvline(best_alpha, color='r', linestyle='--', 
                    label=f'Best alpha = {best_alpha:.4e}\nMSE = {mse_scores[alphas == best_alpha].min():.4f}')

        plt.xlabel('Alpha (Regularization Strength)')
        plt.ylabel(y_axis_label)
        
        title_core = "MSE vs. Alpha"
        if filename_prefix:
            descriptive_name = filename_prefix.replace('_', ' ')
            plt.title(f'{descriptive_name}: {title_core}')
        else:
            plt.title(title_core)
        
        if len(alphas) > 1 and alphas.min() > 0 and alphas.max() / alphas.min() > 100: 
            plt.xscale('log')
            plt.xlabel('Alpha (Regularization Strength) - Log Scale')

        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5) 
        plt.tight_layout()

        base_plot_filename = "mse_vs_alpha.png"
        if filename_prefix:
            final_plot_filename = f"{filename_prefix}_{base_plot_filename}"
        else:
            final_plot_filename = base_plot_filename
        
        plot_path = os.path.join(output_plot_dir, final_plot_filename)

        os.makedirs(output_plot_dir, exist_ok=True)
        
        plt.savefig(plot_path)
        plt.close()
        print(f"    Alpha vs MSE plot saved: {plot_path}")

    except Exception as e:
        print(f"Error generating MSE vs. Alpha plot for {filename_prefix}: {e}")


def plot_model_performance(results_df_final, output_plot_dir):
    """
    Plots R2 Score on the x-axis and RMSE on the y-axis for the test set models.
    Identifies models solely by color using the legend, with no text labels on the plot.

    Args:
        results_df_final (pd.DataFrame): The DataFrame containing the model evaluation results.
        output_plot_dir (str): Directory to save the plot.
    """
    test_results = results_df_final[results_df_final['Model'].str.contains('(Test)')].copy()

    test_results['Model'] = test_results['Model'].str.replace(' (Test)', '', regex=False).str.strip()

    test_results_sorted = test_results.sort_values(by=['R2 Score', 'RMSE'], ascending=[False, True])

    plt.figure(figsize=(16, 12)) 
    scatter_plot = sns.scatterplot(
        data=test_results_sorted,
        x='R2 Score',
        y='RMSE',
        hue='Model',  
        s=250,        
        alpha=0.9,    
        edgecolor='black',
        linewidth=0.7 
    )


    plt.title('Model Performance: R2 Score vs. RMSE (Test Set)', fontsize=18, pad=20)
    plt.xlabel('R2 Score (Higher is Better)', fontsize=14)
    plt.ylabel('RMSE (Lower is Better)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.axvline(x=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    plt.axhline(y=0.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    x_min, x_max = test_results_sorted['R2 Score'].min(), test_results_sorted['R2 Score'].max()
    y_min, y_max = test_results_sorted['RMSE'].min(), test_results_sorted['RMSE'].max()
    plt.xlim(x_min - (x_max - x_min) * 0.15, x_max + (x_max - x_min) * 0.15)
    plt.ylim(y_min - (y_max - y_min) * 0.15, y_max + (y_max - y_min) * 0.15)


    plt.legend(title='Model Type', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.78, 1]) 
    plt.savefig(os.path.join(output_plot_dir, 'model_performance_test_set.png'))

