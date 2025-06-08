from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_housing_data():
    """
    Carga el conjunto de datos de California Housing y realiza una limpieza en 3 pasos:
    1. Elimina el capping de la variable objetivo (MedHouseVal).
    2. Elimina explícitamente el capping de la característica 'HouseAge'.
    3. Elimina outliers generales usando el método del Rango Intercuartílico (IQR).

    Returns:
        pd.DataFrame: El conjunto de datos de California Housing limpio como un DataFrame.
    """
    # Cargar los datos iniciales
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    print("Dataset original cargado.")
    print(f"Tamaño inicial de los datos: {df.shape[0]} muestras")
    print("-" * 50)

    # --- PASO 1: Eliminar el capping en la variable objetivo (MedHouseVal) ---
    initial_rows = df.shape[0]
    df_step1 = df[df['MedHouseVal'] < 5.0].copy()
    rows_after_step1 = df_step1.shape[0]
    print("Paso 1: Limpieza de capping en el target (MedHouseVal < 5)...")
    print(f" -> Tamaño de los datos: {rows_after_step1} muestras")
    print(f" -> Se eliminaron {initial_rows - rows_after_step1} muestras.")
    print("-" * 50)

    # --- PASO 2: Eliminar el capping explícito en 'HouseAge' ---
    # En este dataset, el capping ocurre en el valor máximo.
    house_age_max_value = df_step1['HouseAge'].max()
    print(f"Paso 2: Limpieza de capping en 'HouseAge' (valor máximo = {house_age_max_value})...")
    
    df_step2 = df_step1[df_step1['HouseAge'] < house_age_max_value].copy()
    rows_after_step2 = df_step2.shape[0]
    print(f" -> Tamaño de los datos: {rows_after_step2} muestras")
    print(f" -> Se eliminaron {rows_after_step1 - rows_after_step2} muestras con HouseAge = {house_age_max_value}.")
    print("-" * 50)

    # --- PASO 3: Eliminar outliers restantes de todas las características usando IQR ---
    print("Paso 3: Limpiando outliers generales restantes usando el método IQR...")
    Q1 = df_step2.quantile(0.01)
    Q3 = df_step2.quantile(0.99)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = ~((df_step2 < lower_bound) | (df_step2 > upper_bound)).any(axis=1)
    df_cleaned = df_step2[mask]
    rows_after_step3 = df_cleaned.shape[0]
    
    print(f" -> Tamaño final de los datos: {rows_after_step3} muestras")
    print(f" -> Se eliminaron {rows_after_step2 - rows_after_step3} muestras consideradas outliers.")
    print("-" * 50)

    print("✅ Proceso de carga y limpieza completado.")
    return df_cleaned

def load_raw_data():
    """
    Carga el conjunto de datos de California Housing en su forma cruda,
    sin aplicar ninguna limpieza.

    Returns:
        pd.DataFrame: El conjunto de datos de California Housing original como un DataFrame.
    """
    # Cargar los datos usando scikit-learn con el argumento as_frame=True
    # para obtener directamente un DataFrame de pandas.
    housing = fetch_california_housing(as_frame=True)
    df_raw = housing.frame

    print("Datos crudos cargados exitosamente.")
    print(f"El dataset contiene {df_raw.shape[0]} muestras y {df_raw.shape[1]} columnas.")

    return df_raw