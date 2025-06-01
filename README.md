# LinearRegresion# California Housing Price Prediction

This project aims to predict California housing prices using various regression techniques, including Linear Regression and Ridge Regression. It demonstrates a typical machine learning workflow, from data loading and exploratory data analysis to model training, evaluation, and visualization.

## Project Structure

- `main.py`: The main script to run the entire workflow.
- `data_loader.py`: Handles loading the California Housing dataset.
- `data_analyzer.py`: Contains functions for exploratory data analysis (EDA), including correlation analysis and histograms.
- `model_trainer.py`: Implements the training logic for Linear Regression and Ridge Regression, including hyperparameter tuning with GridSearchCV.
- `model_evaluator.py`: Provides functions to calculate and display key regression evaluation metrics (MSE, MAE, R²).
- `visualizer.py`: Contains all plotting functions to visualize data distributions, model predictions, and residuals.
- `requirements.txt`: Lists all necessary Python packages.
- `README.md`: This file, providing an overview of the project.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RodrigoGoni/LinearRegresion.git
    cd LinearRegresion
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv LR
    # On Windows:
    .\LR\Scripts\activate
    # On macOS/Linux:
    source LR/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

To run the entire analysis and model training pipeline, execute the `main.py` script:

```bash
python main.py