import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import base64

# Title and Introduction
st.title('📈 Do you want to know if your project will end with difficulty?')
st.write("""
The main objective is to generate a comprehensive dashboard, specifically for the Strategic Planning area of AGESIC portfolios, to have indicators available for decision-making.
To achieve this, we analyzed the projects and created models to contribute to this objective. 
Below, based on the EDA (Exploratory Data Analysis), an analysis of 362 variables from 183,281 records of the AGESIC database was conducted.
The analysis revealed that 37,050 records and 16 variables impact the prediction of whether a project will end with difficulty.
We obtained the model that allows predicting whether a project will end with difficulty.
If you want to know this, please select your data:
""")
st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRq2yyslgQq1UyxvTItTShdmw7yiXd5_mcUGA&usqp=CAU')

# Sidebar Information
st.sidebar.image('https://www.gub.uy/agencia-gobierno-electronico-sociedad-informacion-conocimiento/sites/agencia-gobierno-electronico-sociedad-informacion-conocimiento/files/catalogo/iso.png')
st.sidebar.write("""
## Logistic Regression Classification Model
### What is Logistic Regression?
Logistic regression is a technique that helps us predict if something belongs to one of two categories. It can predict if a project will have problems or not.

### How Does it Work?
- **Predicts Probabilities**: Logistic regression calculates the probability of something happening, such as "There is a 70% chance that the project will end with problems."
- **Makes Decisions**: If the calculated probability is greater than or equal to 50%, the model says "Yes" (for example, the project will have problems). If it is less, it says "No."

### Why is it Useful?
It is useful because, based on certain data (like the progress of the project, time, etc.), it can help us make informed decisions and anticipate possible problems.

This model allows you to know if a project will end with difficulty or not. The model metrics are:
- **Accuracy** = 0.86
- **Precision** = 0.90
- **Recall** = 0.80
- **F1-score** = 0.86
""")

# Step 1: User Input
st.write("### Step 1: Is your project on schedule?")
dataset_name = st.selectbox('Enter 1 = If it is delayed / Enter 0 = If it is on schedule', ("0", "1"))

st.write("### Step 2: What is your progress percentage?")
avance_input = st.number_input('Progress Percentage %:', min_value=0, max_value=100)

# Path to the CSV file
file_path = os.path.join('DataModeloRegresion2024.csv')

# Load Data
def load_data(file_path):
    encodings = ['latin1', 'utf-8', 'ISO-8859-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
    return None

# Preprocess the Data
def preprocess_data(df_modelo):
    df_modelo['Avance'] = pd.to_numeric(df_modelo['Avance'], errors='coerce')
    df_modelo.drop(['Programa', 'Proyecto', 'Área', 'Orden', 'Nombre', 'Tipo', 'Área.1', 'Estado', 
                    'Duración plan.', 'Duración', 'Anio', 'Riesgos', 'Tipo_Psp'], axis=1, inplace=True, errors='ignore')
    df_modelo['Avance'].fillna(df_modelo['Avance'].mean(), inplace=True)
    return df_modelo

# Train the Model
def train_model(df_modelo, target_column):
    X = df_modelo.drop([target_column], axis=1)
    y = df_modelo[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Load and Preprocess Data
df_modelo = load_data(file_path)
if df_modelo is not None:
    df_modelo = preprocess_data(df_modelo)

    # Train the Model
    target_column = 'Dificultad'  # Ensure this column exists in your data
    model = train_model(df_modelo, target_column)

    # Make Prediction
    st.write('### Step 3: View the result')
    x_nuevo = pd.DataFrame({'Avance': [avance_input], 'Estado_Cronograma': [dataset_name]})
    resultado = model.predict(x_nuevo)
    if resultado == 0:
        st.write('The project will end WITHOUT difficulty.')
    elif resultado == 1:
        st.write('The project will end WITH difficulty.')
else:
    st.write("Error: The CSV file could not be loaded. Check the path and the file.")

# Download the Model
st.write('### Step 4: Download the model')
csv_exp = df_modelo.to_csv(index=False)
b64 = base64.b64encode(csv_exp.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;modelo_name&gt;.csv**)'


st.image('http://i3campus.co/CONTENIDOS/es-cnbguatemala/content/images/a/a7/buz%25c3%25b3n_de_correo.png')
st.write('For more information, you can write to us at canarysoftware@gmail.com.')


#For run streamlit desde termianl
# 1.Estar en la carpeta de streamlit cd... por ejemplo en este caso es cd src
# luego de estar en la carpetar ejecutar el comando: streamlit run st_app_regresionlog.py
