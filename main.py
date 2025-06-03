import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from linear_regression.data_loader import load_housing_data
from linear_regression.data_analyzer import perform_eda
from linear_regression.model_trainer import (
    train_linear_regression,
    train_ridge_regression,
    train_lasso_regression
)
from linear_regression.model_evaluator import evaluate_model
from linear_regression.data_normalizer import normalize_data
from linear_regression.visualizer import (
    plot_predictions_vs_actual,
    plot_residuals_histogram,
    plot_residuals_vs_predicted,
    plot_ridge_mse_vs_alpha
)

def print_section_header(title, width=80):
    """Prints a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title.upper()} ".center(width, ' '))
    print("=" * width + "\n")

def setup_plot_directories(base_dir='output_plots'):
    """Creates and returns a dictionary of paths for storing plots."""
    paths = {"base": base_dir}
    os.makedirs(base_dir, exist_ok=True)

    paths["eda"] = os.path.join(base_dir, "eda_plots")
    os.makedirs(paths["eda"], exist_ok=True)

    model_types = ["linear", "ridge", "lasso"]
    data_states = ["unprocessed", "std_scaled", "robust_scaled",
                   "std_scaled_log_poly", "robust_scaled_log_poly"]

    for model in model_types:
        model_base_path = os.path.join(base_dir, f'{model}_regression_plots')
        paths[f"{model}_base"] = model_base_path
        os.makedirs(model_base_path, exist_ok=True)
        for state in data_states:
            state_path_key = f"{model}_{state.replace(' ', '_')}" # e.g., linear_unprocessed
            dir_path = os.path.join(model_base_path, state.replace(' ', '_'))
            paths[state_path_key] = dir_path
            os.makedirs(dir_path, exist_ok=True)
    return paths

def train_evaluate_and_log_model(
    model_type, X_train, y_train, X_test, y_test,
    processing_description, # e.g., "Unprocessed", "Standard Scaled Log Poly"
    plot_config, # Dictionary containing relevant plot paths
    results_summary_list,
    column_names_for_scaled_data=None # For converting scaled numpy arrays back to DataFrame for EDA
):
    """
    Trains, evaluates, and logs a single model configuration.
    plot_config requires: 'model_plot_dir', and 'alpha_plot_dir' (for Ridge/Lasso)
    """
    full_model_name = f"{model_type.capitalize()} Regression ({processing_description})"
    print(f"Processing: {full_model_name}")

    model = None
    y_pred_train = None
    y_pred_test = None

    if model_type.lower() == "linear":
        model, y_pred_train, y_pred_test = train_linear_regression(X_train, y_train, X_test)
    elif model_type.lower() in ["ridge", "lasso"]:
        train_func = train_ridge_regression if model_type.lower() == "ridge" else train_lasso_regression
        best_alpha, model, cv_results = train_func(X_train, y_train) # Assumes X_train can be numpy array or DataFrame
        print(f"  Best alpha: {best_alpha:.4f}")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        alpha_plot_filename_prefix = f"{model_type.capitalize()}_{processing_description.replace(' ', '_')}"
        # Ensure plot_ridge_mse_vs_alpha can handle filename_prefix
        plot_ridge_mse_vs_alpha(cv_results, best_alpha, plot_config['alpha_plot_dir'], filename_prefix=alpha_plot_filename_prefix)
        print(f"  Alpha vs MSE plot saved in '{plot_config['alpha_plot_dir']}'.")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    print(f"  Training complete.")

    print(f"\nEvaluating Training Set for {full_model_name}...") # Small separator
    metrics_train = evaluate_model(y_train, y_pred_train, f"{full_model_name} (Train)")
    results_summary_list.append({"Model": f"{full_model_name} (Train)", **metrics_train})

    # Evaluate Test Set
    print(f"\nEvaluating Test Set for {full_model_name}...") # Small separator
    metrics_test = evaluate_model(y_test, y_pred_test, f"{full_model_name} (Test)")
    results_summary_list.append({"Model": f"{full_model_name} (Test)", **metrics_test})
    # Generate plots (primarily for Linear Regression as per original script, but can be extended)
    if model_type.lower() == "linear": # Or if specific plots are desired for other models
        print(f"  Generating diagnostic plots...")
        sanitized_desc = processing_description.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        
        plot_predictions_vs_actual(y_train, y_pred_train, f"{full_model_name}: Preds vs. Actual (Train)",
                                   os.path.join(plot_config['model_plot_dir'], f'preds_vs_actual_train_{sanitized_desc}.png'))
        plot_residuals_histogram(y_train, y_pred_train, f"{full_model_name}: Hist. of Residuals (Train)",
                                 os.path.join(plot_config['model_plot_dir'], f'hist_residuals_train_{sanitized_desc}.png'))
        plot_residuals_vs_predicted(y_pred_train, y_train - y_pred_train, f"{full_model_name}: Residuals vs. Pred (Train)",
                                    os.path.join(plot_config['model_plot_dir'], f'res_vs_pred_train_{sanitized_desc}.png'))

        plot_predictions_vs_actual(y_test, y_pred_test, f"{full_model_name}: Preds vs. Actual (Test)",
                                   os.path.join(plot_config['model_plot_dir'], f'preds_vs_actual_test_{sanitized_desc}.png'))
        plot_residuals_histogram(y_test, y_pred_test, f"{full_model_name}: Hist. of Residuals (Test)",
                                 os.path.join(plot_config['model_plot_dir'], f'hist_residuals_test_{sanitized_desc}.png'))
        plot_residuals_vs_predicted(y_pred_test, y_test - y_pred_test, f"{full_model_name}: Residuals vs. Pred (Test)",
                                    os.path.join(plot_config['model_plot_dir'], f'res_vs_pred_test_{sanitized_desc}.png'))
        print(f"  Diagnostic plots saved in '{plot_config['model_plot_dir']}'.\n")
    else:
        print() # Newline for non-linear models after metrics

def main():
    plot_paths = setup_plot_directories()
    results_summary = []

    print_section_header("DATA LOADING AND INITIAL EDA")
    print("Loading data...")
    df = load_housing_data()
    X_orig = df.drop(columns='MedHouseVal')
    y_orig = df['MedHouseVal']
    print("Data loaded successfully.")

    print("Performing Exploratory Data Analysis (Raw Data)...")
    # Ensure perform_eda is adapted to take output_plot_dir and eda_name_suffix
    perform_eda(X_orig, y_orig, plot_paths["eda"], eda_name_suffix='Raw')
    print("EDA on raw data completed.\n")

    print("Splitting data into training and test sets...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train_raw.shape[0]} samples, Test set size: {X_test_raw.shape[0]} samples")

    # --- Scenario 1: Unprocessed Data ---
    print_section_header("MODELS WITH UNPROCESSED DATA")
    for model_t in ["Linear", "Ridge", "Lasso"]:
        train_evaluate_and_log_model(
            model_t, X_train_raw, y_train, X_test_raw, y_test,
            "Unprocessed",
            {'model_plot_dir': plot_paths[f"{model_t.lower()}_unprocessed"],
             'alpha_plot_dir': plot_paths[f"{model_t.lower()}_unprocessed"]}, # Alpha plots in the same dir
            results_summary
        )

    # --- Scenario 2: StandardScaler Normalized Data (No Log/Poly) ---
    print_section_header("MODELS WITH STANDARDSCALER NORMALIZED DATA")
    X_train_std_scaled, X_test_std_scaled, _ = normalize_data(X_train_raw, X_test_raw, scaler_type='standard')
    X_train_std_scaled_df = pd.DataFrame(X_train_std_scaled, columns=X_train_raw.columns, index=X_train_raw.index) # For EDA
    perform_eda(X_train_std_scaled_df, y_train, plot_paths["eda"], eda_name_suffix='Standard_Scaled')
    print("EDA on StandardScaler normalized data completed.")
    for model_t in ["Linear", "Ridge", "Lasso"]:
        train_evaluate_and_log_model(
            model_t, X_train_std_scaled, y_train, X_test_std_scaled, y_test, # Use scaled numpy arrays
            "Standard Scaled",
            {'model_plot_dir': plot_paths[f"{model_t.lower()}_std_scaled"],
             'alpha_plot_dir': plot_paths[f"{model_t.lower()}_std_scaled"]},
            results_summary
        )

    # --- Scenario 3: RobustScaler Normalized Data (No Log/Poly) ---
    print_section_header("MODELS WITH ROBUSTSCALER NORMALIZED DATA")
    X_train_robust_scaled, X_test_robust_scaled, _ = normalize_data(X_train_raw, X_test_raw, scaler_type='robust')
    X_train_robust_scaled_df = pd.DataFrame(X_train_robust_scaled, columns=X_train_raw.columns, index=X_train_raw.index) # For EDA
    perform_eda(X_train_robust_scaled_df, y_train, plot_paths["eda"], eda_name_suffix='Robust_Scaled')
    print("EDA on RobustScaler normalized data completed.")
    for model_t in ["Linear", "Ridge", "Lasso"]:
        train_evaluate_and_log_model(
            model_t, X_train_robust_scaled, y_train, X_test_robust_scaled, y_test, # Use scaled numpy arrays
            "Robust Scaled",
            {'model_plot_dir': plot_paths[f"{model_t.lower()}_robust_scaled"],
             'alpha_plot_dir': plot_paths[f"{model_t.lower()}_robust_scaled"]},
            results_summary
        )

    # --- Feature Transformation (Log & Polynomial) ---
    print_section_header("FEATURE ENGINEERING: LOGARITHMIC & POLYNOMIAL TRANSFORMATIONS")
    log_transform_cols = ['Population', 'AveOccup', 'MedInc']
    X_train_log = X_train_raw.copy()
    X_test_log = X_test_raw.copy()
    print("Applying Logarithmic transformations...")
    for col in log_transform_cols:
        if col in X_train_log.columns:
            X_train_log[col] = np.log1p(X_train_log[col])
            X_test_log[col] = np.log1p(X_test_log[col])
            print(f"  Applied log1p to: {col}")
    print("Logarithmic transformations completed.")

    poly_features_cols = ['MedInc', 'HouseAge', 'Latitude', 'Longitude'] # MedInc is now log-transformed
    poly_transformer = ColumnTransformer(
        transformers=[('poly', PolynomialFeatures(degree=2, include_bias=False), poly_features_cols)],
        remainder='passthrough', verbose_feature_names_out=True
    )
    print(f"Applying Polynomial Features (degree=2) to: {poly_features_cols}")
    X_train_poly_log_array = poly_transformer.fit_transform(X_train_log)
    X_test_poly_log_array = poly_transformer.transform(X_test_log)
    poly_feature_names = poly_transformer.get_feature_names_out()
    X_train_poly_log = pd.DataFrame(X_train_poly_log_array, columns=poly_feature_names, index=X_train_log.index)
    X_test_poly_log = pd.DataFrame(X_test_poly_log_array, columns=poly_feature_names, index=X_test_log.index)
    print(f"Polynomial features completed. New feature count: {X_train_poly_log.shape[1]}")
    # print("First 5 rows of data after Log and Polynomial features:")
    # print(X_train_poly_log.head())

    # --- Scenario 4: Log + Poly Features with StandardScaler ---
    print_section_header("MODELS WITH LOG+POLY FEATURES & STANDARDSCALER")
    X_train_std_poly_log, X_test_std_poly_log, _ = normalize_data(X_train_poly_log, X_test_poly_log, scaler_type='standard')
    X_train_std_poly_log_df = pd.DataFrame(X_train_std_poly_log, columns=X_train_poly_log.columns, index=X_train_poly_log.index) # For EDA
    perform_eda(X_train_std_poly_log_df, y_train, plot_paths["eda"], eda_name_suffix='Standard_Scaled_Log_Poly')
    print("EDA on Log+Poly + StandardScaler data completed.")
    for model_t in ["Linear", "Ridge", "Lasso"]:
        train_evaluate_and_log_model(
            model_t, X_train_std_poly_log, y_train, X_test_std_poly_log, y_test, # Use scaled numpy arrays
            "Std Scaled Log Poly",
            {'model_plot_dir': plot_paths[f"{model_t.lower()}_std_scaled_log_poly"],
             'alpha_plot_dir': plot_paths[f"{model_t.lower()}_std_scaled_log_poly"]},
            results_summary
        )

    # --- Scenario 5: Log + Poly Features with RobustScaler ---
    print_section_header("MODELS WITH LOG+POLY FEATURES & ROBUSTSCALER")
    X_train_robust_poly_log, X_test_robust_poly_log, _ = normalize_data(X_train_poly_log, X_test_poly_log, scaler_type='robust')
    X_train_robust_poly_log_df = pd.DataFrame(X_train_robust_poly_log, columns=X_train_poly_log.columns, index=X_train_poly_log.index) # For EDA
    perform_eda(X_train_robust_poly_log_df, y_train, plot_paths["eda"], eda_name_suffix='Robust_Scaled_Log_Poly')
    print("EDA on Log+Poly + RobustScaler data completed.")
    for model_t in ["Linear", "Ridge", "Lasso"]:
        train_evaluate_and_log_model(
            model_t, X_train_robust_poly_log, y_train, X_test_robust_poly_log, y_test, # Use scaled numpy arrays
            "Robust Scaled Log Poly",
            {'model_plot_dir': plot_paths[f"{model_t.lower()}_robust_scaled_log_poly"],
             'alpha_plot_dir': plot_paths[f"{model_t.lower()}_robust_scaled_log_poly"]},
            results_summary
        )

    # --- Final Summary ---
    print_section_header("COMPREHENSIVE MODEL EVALUATION SUMMARY")
    results_df = pd.DataFrame(results_summary)
    # Optional: Sort or reorder columns if desired
    # results_df = results_df[['Model', 'MSE', 'RMSE', 'R2 Score', 'MAE']]
    print(results_df.to_string()) # .to_string() ensures the full DataFrame is printed
    
    
    print_section_header("TWO-STAGE MODEL (CLASSIFIER + REGRESSOR)")

    # Use one of your best transformed feature sets
    # For this example, we'll use X_train_std_poly_log and X_test_std_poly_log
    # Make sure these variables hold the pandas DataFrames or numpy arrays from your
    # "MODELS WITH LOG+POLY FEATURES & STANDARDSCALER" section.
    # If they are numpy arrays, ensure column names are not strictly needed by classifier/regressor
    # or convert back to DataFrame if necessary for some model implementations.
    # Assuming X_train_std_poly_log and X_test_std_poly_log are ready.

    # --- Stage 1: Train Classifier (Predict if MedHouseVal == 5) ---
    print("--- Stage 1: Training Classifier (Predicts if MedHouseVal == 5) ---")
    y_class_train = (y_train == 5).astype(int)
    y_class_test = (y_test == 5).astype(int)

    # Using RandomForestClassifier as an example
    # class_weight='balanced' can help if one class is much more frequent
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    
    # Ensure X_train_std_poly_log is suitable (e.g., if it's a DataFrame, direct use is fine)
    # If it's a numpy array from normalize_data, it's also fine.
    classifier.fit(X_train_std_poly_log, y_class_train)

    print("\nClassifier Performance on Test Set:")
    y_class_pred_test_for_eval = classifier.predict(X_test_std_poly_log)
    print("Confusion Matrix (Classifier):")
    print(confusion_matrix(y_class_test, y_class_pred_test_for_eval))
    print("\nClassification Report (Classifier):")
    print(classification_report(y_class_test, y_class_pred_test_for_eval))
    if 0 not in np.unique(y_class_pred_test_for_eval) or 1 not in np.unique(y_class_pred_test_for_eval):
        print("Warning: Classifier predicts only one class. This will affect the two-stage model performance.")


    # --- Stage 2: Train Regression Model (for MedHouseVal < 5) ---
    print("\n--- Stage 2: Training Regression Model (for actual MedHouseVal < 5) ---")

    # Filter original training data for the regressor stage
    uncapped_train_mask = (y_train < 5)
    
    # Use the corresponding rows from the already transformed training features
    X_train_reg_subset = X_train_std_poly_log[uncapped_train_mask]
    y_train_reg_subset = y_train[uncapped_train_mask]

    if len(X_train_reg_subset) == 0:
        print("Warning: No training data available for the regression stage (all y_train values might be >= 5).")
        print("Skipping two-stage model evaluation.")
    elif len(np.unique(y_class_pred_test_for_eval)) < 2 and np.all(y_class_pred_test_for_eval == 1):
        print("Warning: Classifier predicts all instances as capped. Regressor stage will not be used for test predictions.")
        # We can still evaluate, but predictions will all be 5.0
        y_pred_two_stage_test = np.full_like(y_test, 5.0, dtype=float)
        print("\n--- Two-Stage Model Overall Performance on Test Set (Classifier predicts all as 5) ---")
        metrics_two_stage = evaluate_model(y_test, y_pred_two_stage_test, "Two-Stage Model (All Pred as 5) (Test)")
        results_summary.append({"Model": "Two-Stage Model (All Pred as 5) (Test)", **metrics_two_stage})
    else:
        # We'll use Linear Regression for simplicity for the Stage 2 regressor.
        # You could substitute this with your `train_ridge_regression` if you adapt it
        # or directly use a Ridge model with a pre-determined alpha.
        # The `train_linear_regression` function is defined in `linear_regression.model_trainer`
        # It expects X_train, y_train, X_test. For this training part, X_test is a dummy.
        regressor_stage2_model, _, _ = train_linear_regression(
            X_train_reg_subset, y_train_reg_subset, X_train_reg_subset # Using subset as dummy X_test
        )
        print(f"Regression model for uncapped values trained on {len(X_train_reg_subset)} samples.")

        # --- Two-Stage Prediction on Full Test Set ---
        print("\n--- Applying Two-Stage Prediction Logic on Test Set ---")
        y_pred_two_stage_test = np.zeros_like(y_test, dtype=float)

        # Get classifier's predictions for routing on the *actual* test set features
        # This was computed as y_class_pred_test_for_eval earlier for the classifier's own evaluation.
        # Let's rename it for clarity in the loop or re-use.
        test_set_class_predictions = y_class_pred_test_for_eval

        for i in range(len(X_test_std_poly_log)):
            if test_set_class_predictions[i] == 1:  # Classifier predicts MedHouseVal == 5
                y_pred_two_stage_test[i] = 5.0
            else:  # Classifier predicts MedHouseVal < 5
                # Ensure the sample passed to predict is correctly shaped (2D array for scikit-learn)
                if isinstance(X_test_std_poly_log, pd.DataFrame):
                    sample_to_predict = X_test_std_poly_log.iloc[[i]]
                else: # Assuming numpy array
                    sample_to_predict = X_test_std_poly_log[i].reshape(1, -1)
                
                prediction = regressor_stage2_model.predict(sample_to_predict)[0]
                
                # Practical clipping: regressor is trained on <5, so its predictions should reflect that.
                if prediction >= 5.0:
                    prediction = 4.999 # Clip to just below 5
                if prediction < 0:
                    prediction = 0.0 # Floor at 0
                y_pred_two_stage_test[i] = prediction
        
        print("Two-stage predictions generated for the test set.")

        # --- Evaluate Two-Stage Model ---
        print("\n--- Two-Stage Model Overall Performance on Test Set ---")
        metrics_two_stage = evaluate_model(y_test, y_pred_two_stage_test, "Two-Stage Model (Test)")
        results_summary.append({"Model": "Two-Stage Model (Classifier + Regressor) (Test)", **metrics_two_stage})

        # --- Optional: Detailed Evaluation on Subsets of Test Data ---
        test_uncapped_mask = (y_test < 5)
        test_capped_mask = (y_test == 5)

        if np.sum(test_uncapped_mask) > 0:
            print("\n--- Two-Stage Model Performance on ACTUAL Uncapped Test Data (y_test < 5) ---")
            evaluate_model(
                y_test[test_uncapped_mask],
                y_pred_two_stage_test[test_uncapped_mask],
                "Two-Stage Model (Test, actual y < 5)"
            )
        
        if np.sum(test_capped_mask) > 0:
            print("\n--- Two-Stage Model Performance on ACTUAL Capped Test Data (y_test == 5) ---")
            evaluate_model(
                y_test[test_capped_mask],
                y_pred_two_stage_test[test_capped_mask],
                "Two-Stage Model (Test, actual y == 5)"
            )

    # ... (rest of your main function, like printing the final results_summary DataFrame) ...
    
    print_section_header("COMPREHENSIVE MODEL EVALUATION SUMMARY (WITH TWO-STAGE)")
    results_df_final = pd.DataFrame(results_summary)
    print(results_df_final.to_string())

if __name__ == "__main__":
    main()