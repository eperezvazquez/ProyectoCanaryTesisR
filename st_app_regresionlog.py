#STREAMLIT REGRESION LOGISTICA
import pandas as pd
import numpy as np 
import altair as alt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
df_modelo = pd.read_csv('https://raw.githubusercontent.com/eperezvazquez/ProyectoCanaryTesisR/main/src/DataModeloRegresion.csv', engine='python')
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

st.image('http://i3campus.co/CONTENIDOS/es-cnbguatemala/content/images/a/a7/buz%25c3%25b3n_de_correo.png')
st.write('Por mas informacion nos puede escribir canarysoftware@gmail.com.')
