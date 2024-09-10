#STREAMLIT REGRESION LOGISTICA
import pandas as pd
import numpy as np
import base64 
import streamlit as st

st.title('📈 ¿Quiere saber si su proyecto termina con dificultad?')
"""
El objetivo general es poder generar un cuadro de mando integral, del área de Planificación Estratégica en particular de los portafolios de AGESIC, donde permita tener indicadores a la hora de la toma de decisiones. 
Para ello analizamos los proyectos y genereamos modelos que contribuyan en ese sentido. 
Debajo en base al EDA, analisis de exploracion de datos donde se analizaron 362 variables de 12952 registros de la base de datos de AGESIC, SIGES con una precison del 93% se obtuvo el modelo debajo donde permite predecir si un proyecto terminara con dificultad o no.
Si ustedes quiere saber eso por favor selecione sus datos:
"""
st.image ('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRq2yyslgQq1UyxvTItTShdmw7yiXd5_mcUGA&usqp=CAU')

st.sidebar.image('https://www.gub.uy/agencia-gobierno-electronico-sociedad-informacion-conocimiento/sites/agencia-gobierno-electronico-sociedad-informacion-conocimiento/files/catalogo/iso.png')
st.sidebar.write("""
## Modelo de Clasificacion de Regresion logistica que nos permite saber si un proyecto terminara con dificultad o no.
Asimismo los datos del modelo son:
#
Accuracy = 0.93, La exactitud (accuracy) mide el porcentaje de casos que el modelo ha acertado
#
Precision = 0.94, Precision nos da la calidad de la predicción: ¿qué porcentaje de los que hemos dicho que son la clase positiva, en realidad lo son?
#
Recall =  0.93, nos da la cantidad: ¿qué porcentaje de la clase positiva hemos sido capaces de identificar?
#
f1-score = 0.93, combina Precision y Recall en una sola medida
""")

"""
### Paso 1: ¿Su proyecto esta en fecha? 
Recuerde que por lineamiento de AGESIC hasta el 20% de desvio se considera en fecha.
"""
dataset_name = st.selectbox('Ingrese 1= Si esta atrazado/Ingrese 0= Si esta en fecha',("0","1")) 


###

"""
### Paso2: ¿Cual es su porcentaje de avance? 
Ingrese el porcentaje de avance en numeros.
"""
avance_input = st.number_input('Porcentaje de Avance %:',min_value = 0, max_value = 100)
 

"""
### Paso 3: Se visualiza el resultado
Debajo se muestra el resultado de la prediccion, una vez ingresadas las variables.
"""
#MODELO DE REGRESION
df_modelo = pd.read_csv('https://raw.githubusercontent.com/eperezvazquez/ProyectoCanaryTesisR/main/src/DataModeloRegresion2024.csv', engine='python')
#Eliminan las filas de esos registros
indexNames = df_modelo[df_modelo['Padre']==0].index
# Delete these row indexes from dataFrame
df_modelo.drop(indexNames,inplace=True)
from sklearn.model_selection import train_test_split 
train, test = train_test_split(df_modelo, test_size = 0.20, shuffle = False)
#Eliminan las filas de esos registros
indexNames = train[train['Avance'].isnull()].index
# Delete these row indexes from dataFrame
train.drop(indexNames,inplace=True)
#Eliminan las filas de esos registros
indexNames = train[train['EstadoCronograma'].isnull()].index
# Delete these row indexes from dataFrame
train.drop(indexNames,inplace=True)
train.drop(['Tipo_Presupuesto','Termina en presupuesto','Programa','Proyecto','Área','Orden','Padre','Nombre','Área.1','Tipo','Inicio plan.','Fin plan.','Inicio','Fin','Duración plan.','Duración','Cantidad_Riesgos','Tipo_Riesgo','Anio'],axis=1,inplace=True)
X = train.drop(['Dificultad'], axis=1)
y = train['Dificultad']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
# Instantiate Logistic Regression
model = LogisticRegression()
# Fit the data
model.fit(X_train, y_train)
#INGRESO LOS VALORES AL MODELO
st.write('Resultados del modelo:')
x_nuevo = pd.DataFrame({'Avance':[avance_input],'EstadoCronograma': [dataset_name]})
resultado=model.predict(x_nuevo)
if resultado == 0:
   st.write('Proyecto terminara SIN dificultad: {}'.format(resultado))
elif resultado == 1:
   st.write('Proyecto terminara CON dificultad: {}'.format(resultado))
elif resultado=='' :
    st.write('Ingrese los valores porque por defecto es 0,0 un proyecto con dificultad')
else:
    st.write('Ingrese los valores porque por defecto es 0,0 un proyecto con dificultad')

st.write('Pasamos al paso 4 donde bajamos el modelo')   
"""
### Paso 4: Se permite descargar el modelo
"""
"""
El siguiente enlace le permite descargar el pronóstico recién creado a su computadora para su posterior análisis y uso.
"""
csv_exp = df_modelo.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;modelo_name&gt;.csv**)'
st.markdown(href, unsafe_allow_html=True)

st.image('data:image/jpeg;base64,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')
st.write('Por mas informacion nos puede escribir canarysoftware@gmail.com.')
