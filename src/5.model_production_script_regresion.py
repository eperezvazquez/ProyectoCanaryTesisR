#Puesta en produccion modulo regression
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import pickle
import os

# Ruta del archivo CSV
file_path = os.path.join('src', 'DataModeloRegresion2024.csv')

# Función para cargar el archivo CSV con manejo de errores y múltiples codificaciones
def load_data(file_path):
    # Verificar si el archivo existe
    if not os.path.exists(file_path):
        print(f"Error: El archivo {file_path} no existe.")
        return None
    
    # Lista de codificaciones a probar
    encodings = ['latin1', 'utf-8', 'ISO-8859-1']
    
    # Intentar cargar el archivo con diferentes codificaciones
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Archivo cargado correctamente con codificación: {encoding}")
            return df
        except UnicodeDecodeError:
            print(f"Error de codificación con {encoding}. Intentando con la siguiente...")
        except Exception as e:
            print(f"Error al cargar el archivo con {encoding}: {e}")
            return None
    
    # Si no se pudo cargar con ninguna codificación
    print("Error: No se pudo cargar el archivo con las codificaciones probadas.")
    return None

# Preprocess the data
def preprocess_data(df_modelo):
    # Convertir las columnas a float64, forzando la conversión de cualquier valor no numérico a NaN
    df_modelo['Avance'] = pd.to_numeric(df_modelo['Avance'], errors='coerce')

    # Eliminar columnas específicas del DataFrame
    df_modelo.drop([
        'Programa',
        'Proyecto',
        'Área',
        'Orden',
        'Nombre',
        'Tipo',
        'Área.1',
        'Estado',
        'Duración plan.',
        'Duración',
        'Anio',
        'Riesgos',
        'Tipo_Psp'
    ], axis=1, inplace=True, errors='ignore')  # 'errors=ignore' to avoid errors if columns don't exist
    
    # Replace missing values in the 'Avance' column with the mean of the column
    if 'Avance' in df_modelo.columns:
        df_modelo['Avance'].fillna(df_modelo['Avance'].mean(), inplace=True)
    
    return df_modelo

# Train the classification model with hyperparameter tuning
def train_model(df_modelo, target_column):
    # Define features and target
    X = df_modelo.drop([target_column], axis=1)
    y = df_modelo[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the hyperparameter grid to search with valid combinations
    param_grid = [
        {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]},
        {'solver': ['newton-cg', 'lbfgs'], 'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100]}
    ]
    
    # Initialize the model
    model = LogisticRegression(max_iter=1000)
    
    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model using accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    
    return best_model

# Save the trained model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")

# Main function to run the script
if __name__ == "__main__":
    # Load the data
    df_modelo = load_data(file_path)
    
    # Check if the data is loaded successfully
    if df_modelo is not None:
        # Preprocess the data
        df_modelo = preprocess_data(df_modelo)
        
        # Define the target column (replace 'Dificultad' with the actual target column name)
        target_column = 'Dificultad'  # Asegúrate de que 'Dificultad' sea adecuada para clasificación
        
        # Train the model with hyperparameter tuning
        model = train_model(df_modelo, target_column)
        
        # Save the model
        save_model(model, 'classification_model_optimized.pkl')
