import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
 
def perform_eda(data_df, target_series, output_plot_dir, eda_name_suffix=""):
    """
    Performs Exploratory Data Analysis (EDA) and saves consolidated plots.

    This revised function creates a single figure for all feature histograms and 
    a single figure for all feature-vs-target scatter plots, instead of one file per plot.

    Args:
        data_df (pd.DataFrame): DataFrame containing the features.
        target_series (pd.Series): Series containing the target variable.
        output_plot_dir (str): Directory where the generated plots will be saved.
        eda_name_suffix (str, optional): Suffix for filenames to distinguish EDA runs
                                         (e.g., "Raw", "Standard_Scaled"). Defaults to "".
    """
    print(f"  Iniciando EDA para: {eda_name_suffix if eda_name_suffix else 'el dataset'}")
    os.makedirs(output_plot_dir, exist_ok=True)

    # Helper function to create consistent filenames and titles
    def get_names(base_name, suffix):
        title = base_name.replace('_', ' ').title()
        filename = f"{base_name.lower()}"
        if suffix:
            title += f" ({suffix})"
            filename += f"_{suffix}"
        return title, f"{filename}.png"

    # 1. Correlation Matrix (remains a single plot)
    if not data_df.empty:
        plt.figure(figsize=(16, 12))
        correlation_matrix = data_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        
        title, filename = get_names("Matriz_de_Correlacion_de_Caracteristicas", eda_name_suffix)
        plt.title(title, fontsize=18)
        
        plot_path = os.path.join(output_plot_dir, filename)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"    Guardado: {plot_path}")

    # 2. Target Variable Distribution (remains a single plot)
    if not target_series.empty:
        plt.figure(figsize=(8, 6))
        sns.histplot(target_series, kde=True)
        
        title, filename = get_names("Distribucion_del_Target", eda_name_suffix)
        plt.title(title, fontsize=16)
        
        plot_path = os.path.join(output_plot_dir, filename)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"    Guardado: {plot_path}")

    # 3. Histograms of all features in one figure
    if not data_df.empty:
        print("  Generando figura combinada de histogramas...")
        features = data_df.columns
        num_features = len(features)
        
        # Define grid layout
        n_cols = 4  
        n_rows = math.ceil(num_features / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten() # Flatten to 1D array for easy iteration

        for i, feature in enumerate(features):
            sns.histplot(data_df[feature], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribución de {feature}', fontsize=12)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')

        # Hide unused subplots
        for i in range(num_features, len(axes)):
            axes[i].set_visible(False)

        main_title, filename = get_names("Histogramas_de_Caracteristicas", eda_name_suffix)
        fig.suptitle(main_title, fontsize=20)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
        plot_path = os.path.join(output_plot_dir, filename)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"    Guardado: {plot_path}")

    # 4. Scatter plots of all features vs. target in one figure
    if not data_df.empty and not target_series.empty:
        print("  Generando figura combinada de gráficos de dispersión...")
        features = data_df.columns
        num_features = len(features)
        
        # Define grid layout
        n_cols = 4
        n_rows = math.ceil(num_features / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten()

        for i, feature in enumerate(features):
            sns.scatterplot(x=data_df[feature], y=target_series, ax=axes[i], alpha=0.5)
            axes[i].set_title(f'{feature} vs. Target', fontsize=12)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')

        # Hide unused subplots
        for i in range(num_features, len(axes)):
            axes[i].set_visible(False)
        
        main_title, filename = get_names("Caracteristicas_vs_Target", eda_name_suffix)
        fig.suptitle(main_title, fontsize=20)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = os.path.join(output_plot_dir, filename)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"    Guardado: {plot_path}")

    print(f"  EDA para {eda_name_suffix if eda_name_suffix else 'el dataset'} completado.")