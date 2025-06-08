import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from linear_regression.data_loader import load_housing_data, load_raw_data
from linear_regression.data_analyzer import perform_eda
from linear_regression.model_trainer import (
    train_linear_regression,
    train_ridge_regression,
    train_lasso_regression
)
from linear_regression.model_evaluator import evaluate_model
from linear_regression.data_normalizer import normalize_data
from linear_regression.visualizer import (
    plot_diagnostic_subplots,
    plot_ridge_mse_vs_alpha,
    plot_model_performance
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
                   "std_scaled_log_poly", "robust_scaled_log_poly","robust_scaled_RAW"]

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
    processing_description,
    plot_config,
    results_summary_list,
    column_names_for_scaled_data=None
):
    """
    Trains, evaluates, and logs a single model configuration.
    This version creates a single consolidated figure for diagnostic plots.
    
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
        best_alpha, model, cv_results = train_func(X_train, y_train)
        print(f"  Best alpha: {best_alpha:.4f}")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        alpha_plot_filename_prefix = f"{model_type.capitalize()}_{processing_description.replace(' ', '_')}"
        plot_ridge_mse_vs_alpha(cv_results, best_alpha, plot_config['alpha_plot_dir'], filename_prefix=alpha_plot_filename_prefix)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    print(f"  Training complete.")

    print(f"\nEvaluating Training Set for {full_model_name}...")
    metrics_train = evaluate_model(y_train, y_pred_train, f"{full_model_name} (Train)")
    results_summary_list.append({"Model": f"{full_model_name} (Train)", **metrics_train})

    print(f"\nEvaluating Test Set for {full_model_name}...")
    metrics_test = evaluate_model(y_test, y_pred_test, f"{full_model_name} (Test)")
    results_summary_list.append({"Model": f"{full_model_name} (Test)", **metrics_test})
    
    # *** REFACTORED PLOTTING SECTION ***
    if model_type.lower() == "linear":
        print(f"  Generating consolidated diagnostic plot...")
        sanitized_desc = processing_description.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        
        # Define the single save path for the consolidated plot
        save_path = os.path.join(plot_config['model_plot_dir'], f'diagnostic_plots_{sanitized_desc}.png')
        
        # Call the new function to create and save the 2x3 grid of plots
        plot_diagnostic_subplots(
            y_train_true=y_train,
            y_train_pred=y_pred_train,
            y_test_true=y_test,
            y_test_pred=y_pred_test,
            model_name=full_model_name,
            save_path=save_path
        )
        print(f"  Diagnostic plots saved in a single figure: '{save_path}'.\n")
    else:
        print(f"  Generating consolidated diagnostic plot...")
        sanitized_desc = processing_description.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        
        # Define the single save path for the consolidated plot
        save_path = os.path.join(plot_config['model_plot_dir'], f'diagnostic_plots_{sanitized_desc}.png')
        plot_diagnostic_subplots(
            y_train_true=y_train,
            y_train_pred=y_pred_train,
            y_test_true=y_test,
            y_test_pred=y_pred_test,
            model_name=full_model_name,
            save_path=save_path)
        print() # Maintain spacing for non-linear models
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
    X_train, X_test, y_train, y_test = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]} samples, Test set size: {X_test.shape[0]} samples")

    # --- Scenario 1: Unprocessed Data ---
    print_section_header("MODELS WITH UNSCALED DATA")
    for model_t in ["Linear", "Ridge", "Lasso"]:
        train_evaluate_and_log_model(
            model_t, X_train, y_train, X_test, y_test,
            "Unscaled",
            {'model_plot_dir': plot_paths[f"{model_t.lower()}_unprocessed"],
             'alpha_plot_dir': plot_paths[f"{model_t.lower()}_unprocessed"]},
            results_summary
        )

    # --- Scenario 2: StandardScaler Normalized Data (No Log/Poly) ---
    print_section_header("MODELS WITH STANDARDSCALER NORMALIZED DATA")
    X_train_std_scaled, X_test_std_scaled, _ = normalize_data(X_train, X_test, scaler_type='standard')
    X_train_std_scaled_df = pd.DataFrame(X_train_std_scaled, columns=X_train.columns, index=X_train.index) 
    perform_eda(X_train_std_scaled_df, y_train, plot_paths["eda"], eda_name_suffix='Standard_Scaled')
    print("EDA on StandardScaler normalized data completed.")
    for model_t in ["Linear", "Ridge", "Lasso"]:
        train_evaluate_and_log_model(
            model_t, X_train_std_scaled, y_train, X_test_std_scaled, y_test,
            "Standard Scaled",
            {'model_plot_dir': plot_paths[f"{model_t.lower()}_std_scaled"],
             'alpha_plot_dir': plot_paths[f"{model_t.lower()}_std_scaled"]},
            results_summary
        )

    # --- Scenario 3: RobustScaler Normalized Data (No Log/Poly) ---
    print_section_header("MODELS WITH ROBUSTSCALER NORMALIZED DATA")
    X_train_robust_scaled, X_test_robust_scaled, _ = normalize_data(X_train, X_test, scaler_type='robust')
    X_train_robust_scaled_df = pd.DataFrame(X_train_robust_scaled, columns=X_train.columns, index=X_train.index) 
    perform_eda(X_train_robust_scaled_df, y_train, plot_paths["eda"], eda_name_suffix='Robust_Scaled')
    print("EDA on RobustScaler normalized data completed.")
    for model_t in ["Linear", "Ridge", "Lasso"]:
        train_evaluate_and_log_model(
            model_t, X_train_robust_scaled, y_train, X_test_robust_scaled, y_test,
            "Robust Scaled",
            {'model_plot_dir': plot_paths[f"{model_t.lower()}_robust_scaled"],
             'alpha_plot_dir': plot_paths[f"{model_t.lower()}_robust_scaled"]},
            results_summary
        )

    # --- Scenario 4: RobustScaler Normalized RAW Data (No Log/Poly) ---
    print_section_header("MODELS WITH ROBUSTSCALER NORMALIZED RAW DATA")
    print("Splitting data into training and test sets...")
    df_raw = load_raw_data()
    X_orig_raw = df_raw.drop(columns='MedHouseVal')
    y_orig_raw = df_raw['MedHouseVal']
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_orig_raw, y_orig_raw, test_size=0.2, random_state=42
    )
    X_train_robust_scaled, X_test_robust_scaled, _ = normalize_data(X_train_raw, X_test_raw, scaler_type='robust')
    X_train_robust_scaled_df = pd.DataFrame(X_train_robust_scaled, columns=X_train_raw.columns, index=X_train_raw.index) 
    perform_eda(X_train_robust_scaled_df, y_train_raw, plot_paths["eda"], eda_name_suffix='Robust_Scaled_RAW')
    print("EDA on RobustScaler normalized RAW data completed.")
    for model_t in ["Linear", "Ridge", "Lasso"]:
        train_evaluate_and_log_model(
            model_t, X_train_robust_scaled, y_train_raw, X_test_robust_scaled, y_test_raw,
            "Robust_Scaled_RAW",
            {'model_plot_dir': plot_paths[f"{model_t.lower()}_robust_scaled_RAW"],
             'alpha_plot_dir': plot_paths[f"{model_t.lower()}_robust_scaled_RAW"]},
            results_summary
        )
    # --- Feature Transformation (Log & Polynomial) ---
    print_section_header("FEATURE ENGINEERING: LOGARITHMIC & POLYNOMIAL TRANSFORMATIONS")
    log_transform_cols = ['Population', 'AveOccup', 'MedInc']
    X_train_log = X_train.copy()
    X_test_log = X_test.copy()
    print("Applying Logarithmic transformations...")
    for col in log_transform_cols:
        if col in X_train_log.columns:
            X_train_log[col] = np.log1p(X_train_log[col])
            X_test_log[col] = np.log1p(X_test_log[col])
            print(f"  Applied log1p to: {col}")
    print("Logarithmic transformations completed.")

    poly_features_cols = ['MedInc', 'HouseAge', 'Latitude', 'Longitude']
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
    print("First 5 rows of data after Log and Polynomial features:")

    # --- Scenario 4: Log + Poly Features with StandardScaler ---
    print_section_header("MODELS WITH LOG+POLY FEATURES & STANDARDSCALER")
    X_train_std_poly_log, X_test_std_poly_log, _ = normalize_data(X_train_poly_log, X_test_poly_log, scaler_type='standard')
    X_train_std_poly_log_df = pd.DataFrame(X_train_std_poly_log, columns=X_train_poly_log.columns, index=X_train_poly_log.index) 
    perform_eda(X_train_std_poly_log_df, y_train, plot_paths["eda"], eda_name_suffix='Standard_Scaled_Log_Poly')
    print("EDA on Log+Poly + StandardScaler data completed.")
    for model_t in ["Linear", "Ridge", "Lasso"]:
        train_evaluate_and_log_model(
            model_t, X_train_std_poly_log, y_train, X_test_std_poly_log, y_test,
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
            model_t, X_train_robust_poly_log, y_train, X_test_robust_poly_log, y_test, 
            "Robust Scaled Log Poly",
            {'model_plot_dir': plot_paths[f"{model_t.lower()}_robust_scaled_log_poly"],
             'alpha_plot_dir': plot_paths[f"{model_t.lower()}_robust_scaled_log_poly"]},
            results_summary
        )
    
    
    print_section_header("TWO-STAGE MODEL (CLASSIFIER + REGRESSOR)")

    # --- Stage 1: Train Classifier (Predict if MedHouseVal == 5) ---
    print("--- Stage 1: Training Classifier (Predicts if MedHouseVal == 5) ---")
    y_class_train = (y_train == 5).astype(int)
    y_class_test = (y_test == 5).astype(int)

    # Using RandomForestClassifier as an example
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    
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
        regressor_stage2_model, _, _ = train_linear_regression(
            X_train_reg_subset, y_train_reg_subset, X_train_reg_subset 
        )
        print(f"Regression model for uncapped values trained on {len(X_train_reg_subset)} samples.")

        # --- Two-Stage Prediction on Full Test Set ---
        print("\n--- Applying Two-Stage Prediction Logic on Test Set ---")
        y_pred_two_stage_test = np.zeros_like(y_test, dtype=float)

        test_set_class_predictions = y_class_pred_test_for_eval

        for i in range(len(X_test_std_poly_log)):
            if test_set_class_predictions[i] == 1:  # Classifier predicts MedHouseVal == 5
                y_pred_two_stage_test[i] = 5.0
            else: 
                if isinstance(X_test_std_poly_log, pd.DataFrame):
                    sample_to_predict = X_test_std_poly_log.iloc[[i]]
                else:
                    sample_to_predict = X_test_std_poly_log[i].reshape(1, -1)
                
                prediction = regressor_stage2_model.predict(sample_to_predict)[0]
                
                if prediction >= 5.0:
                    prediction = 4.999 
                if prediction < 0:
                    prediction = 0.0 
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
    results_df_final.sort_values(by='R2 Score', ascending=False, inplace=True)
    plot_model_performance(results_df_final, '/home/rodrigo/Desktop/maestria/LinearRegresion/output_plots')
    print(results_df_final.to_string())

if __name__ == "__main__":
    main()