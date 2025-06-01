import os
from linear_regression.data_loader import load_housing_data
from linear_regression.data_analyzer import perform_eda
from linear_regression.model_trainer import train_linear_regression, train_ridge_regression
from linear_regression.model_evaluator import evaluate_model
from linear_regression.visualizer import (
    plot_predictions_vs_actual,
    plot_residuals_histogram,
    plot_residuals_vs_predicted,
    plot_ridge_mse_vs_alpha
)
from sklearn.model_selection import train_test_split


def main():
    # Define output directories
    output_base_dir = 'output_plots'
    regression_plots_dir = os.path.join(
        output_base_dir, 'linear_regression_plots')
    ridge_plots_dir = os.path.join(output_base_dir, 'ridge_regression_plots')
    os.makedirs(regression_plots_dir, exist_ok=True)
    os.makedirs(ridge_plots_dir, exist_ok=True)

    print("Loading data...")
    df = load_housing_data()
    print("Data loaded successfully.\n")

    X = df.drop(columns='MedHouseVal')
    y = df['MedHouseVal']

    print("Performing Exploratory Data Analysis...")
    perform_eda(X, y, output_base_dir)
    print("EDA completed and plots saved.\n")

    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples\n")

    # --- Linear Regression ---
    print("Training Linear Regression model...")
    linear_model, y_pred_train_lr, y_pred_test_lr = train_linear_regression(
        X_train, y_train, X_test
    )
    print("Linear Regression model trained.\n")

    print("Evaluating Linear Regression model on training set...")
    evaluate_model(y_train, y_pred_train_lr, "Linear Regression (Training)")
    print("\nEvaluating Linear Regression model on test set...")
    evaluate_model(y_test, y_pred_test_lr, "Linear Regression (Test)")
    print()

    print("Generating Linear Regression plots...")
    plot_predictions_vs_actual(y_train, y_pred_train_lr, "Regresión Lineal: Predicciones vs. Valores Reales (Entrenamiento)",
                               os.path.join(regression_plots_dir, 'predicciones_vs_reales_train.png'))
    plot_residuals_histogram(y_train, y_pred_train_lr, "Histograma de Residuos (Entrenamiento)",
                             os.path.join(regression_plots_dir, 'histograma_residuos_train.png'))
    plot_residuals_vs_predicted(y_pred_train_lr, y_train - y_pred_train_lr, "Residuos vs. Valores Predichos (Entrenamiento)",
                                os.path.join(regression_plots_dir, 'residuos_vs_predichos_train.png'))

    plot_predictions_vs_actual(y_test, y_pred_test_lr, "Regresión Lineal: Predicciones vs. Valores Reales (Test)",
                               os.path.join(regression_plots_dir, 'predicciones_vs_reales_test.png'))
    plot_residuals_histogram(y_test, y_pred_test_lr, "Histograma de Residuos (Test)",
                             os.path.join(regression_plots_dir, 'histograma_residuos_test.png'))
    plot_residuals_vs_predicted(y_pred_test_lr, y_test - y_pred_test_lr, "Residuos vs. Valores Predichos (Test)",
                                os.path.join(regression_plots_dir, 'residuos_vs_predichos_test.png'))
    print(f"Linear Regression plots saved in '{regression_plots_dir}'.\n")

    # --- Ridge Regression ---
    print("Training Ridge Regression model with GridSearchCV...")
    best_alpha, best_ridge_model, cv_results = train_ridge_regression(
        X_train, y_train
    )
    print(f"Best alpha found: {best_alpha:.4f}\n")

    print("Evaluating Ridge Regression model on training set...")
    y_pred_train_ridge = best_ridge_model.predict(X_train)
    evaluate_model(y_train, y_pred_train_ridge, "Ridge Regression (Training)")
    print("\nEvaluating Ridge Regression model on test set...")
    y_pred_test_ridge = best_ridge_model.predict(X_test)
    evaluate_model(y_test, y_pred_test_ridge, "Ridge Regression (Test)")
    print()

    print("Generating Ridge Regression plots...")
    plot_ridge_mse_vs_alpha(cv_results, best_alpha, ridge_plots_dir)
    print(f"Ridge Regression plots saved in '{ridge_plots_dir}'.")


if __name__ == "__main__":
    main()
