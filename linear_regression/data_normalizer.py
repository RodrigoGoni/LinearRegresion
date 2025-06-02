from sklearn.preprocessing import StandardScaler
import pandas as pd


def normalize_data(X_train, X_test):
    """
    Normalizes the training and test data using StandardScaler.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.

    Returns:
        tuple: Scaled X_train, Scaled X_test, and the scaler object.
    """
    scaler = StandardScaler()

    # Fit the scaler only on the training data and transform both training and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame with original column names
    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index)

    print("Data normalized using StandardScaler.")
    return X_train_scaled, X_test_scaled, scaler
