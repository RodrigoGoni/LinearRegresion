import seaborn as sns
import matplotlib.pyplot as plt
import os


def perform_eda(X, y, output_dir, name=''):
    """
    Performs exploratory data analysis, including correlation analysis
    and histograms, and saves the plots.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target Series.
        output_dir (str): Directory to save the plots.
    """
    print("\n--- Correlation Analysis ---")
    correlations = X.corrwith(y).sort_values(ascending=False)
    for feature_name, correlation_value in correlations.items():
        print(f"{feature_name}: {correlation_value:.4f}")

    plt.figure(figsize=(10, 6))
    sns.barplot(x=correlations.values,
                y=correlations.index, palette='coolwarm')
    plt.title('Correlación lineal con MedHouseVal')
    plt.xlabel('Coeficiente de correlación')
    plt.ylabel('Atributos')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_correlation.jpg"))
    plt.close()
    print(
        f"Correlation plot saved to {os.path.join(output_dir, '_correlation.jpg')}")

    print("\n--- Histograms of Features ---")
    X.hist(bins=50, figsize=(20, 15))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_histograms.jpg"))
    plt.close()
    print(
        f"Histograms saved to {os.path.join(output_dir,'_histograms.jpg')}")
