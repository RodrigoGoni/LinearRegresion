import seaborn as sns
import matplotlib.pyplot as plt
import os


def perform_eda(data_df, target_series, output_plot_dir, eda_name_suffix=""): # NEW SIGNATURE
    """
    Performs Exploratory Data Analysis and saves plots.

    Args:
        data_df (pd.DataFrame): DataFrame containing the features.
        target_series (pd.Series): Series containing the target variable.
        output_plot_dir (str): Directory to save the generated plots.
        eda_name_suffix (str, optional): Suffix to append to filenames 
                                         to distinguish EDA plots (e.g., "Raw", "Normalized").
                                         Defaults to "".
    """
    print(f"  Starting EDA for: {eda_name_suffix if eda_name_suffix else 'dataset'}")

    # Ensure output directory exists (though main.py should have created it)
    os.makedirs(output_plot_dir, exist_ok=True)

    # Example 1: Correlation Matrix of features
    if not data_df.empty:
        plt.figure(figsize=(12, 10))
        correlation_matrix = data_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        title_str = "Feature Correlation Matrix"
        filename_base = "feature_correlation_matrix"
        if eda_name_suffix:
            title_str += f" ({eda_name_suffix})"
            filename_base += f"_{eda_name_suffix}"
        
        plt.title(title_str)
        plot_path = os.path.join(output_plot_dir, f"{filename_base}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"    Saved: {plot_path}")

    # Example 2: Distribution of the target variable
    if not target_series.empty:
        plt.figure(figsize=(8, 6))
        sns.histplot(target_series, kde=True)
        title_str = "Target Variable Distribution"
        filename_base = "target_distribution"
        if eda_name_suffix: # Suffix also applies if EDA is on training target
            title_str += f" ({eda_name_suffix})" # Though usually target dist is on y_orig
            filename_base += f"_{eda_name_suffix}"

        plt.title(title_str)
        plot_path = os.path.join(output_plot_dir, f"{filename_base}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"    Saved: {plot_path}")

    # Example 3: Scatter plots of some features against the target
    # Ensure data_df and target_series can be combined (e.g., have compatible indices if needed)
    # For simplicity, let's pick a few features if available
    if not data_df.empty and not target_series.empty:
        # Make sure indices align if they come from different train/test splits potentially
        # This example assumes they are aligned or data_df is from X_orig when target_series is y_orig
        temp_df_for_scatter = data_df.copy()
        temp_df_for_scatter['target'] = target_series.values # Ensure alignment

        features_to_plot = data_df.columns[:min(3, len(data_df.columns))] # Plot first 3 features
        for feature in features_to_plot:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=temp_df_for_scatter[feature], y=temp_df_for_scatter['target'])
            title_str = f"{feature} vs. Target"
            filename_base = f"scatter_{feature}_vs_target"
            if eda_name_suffix:
                title_str += f" ({eda_name_suffix})"
                filename_base += f"_{eda_name_suffix}"
            
            plt.title(title_str)
            plot_path = os.path.join(output_plot_dir, f"{filename_base}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"    Saved: {plot_path}")

    print(f"  EDA for {eda_name_suffix if eda_name_suffix else 'dataset'} completed.")
