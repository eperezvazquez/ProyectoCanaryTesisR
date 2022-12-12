#PASAMOS A PRODUCCION EL MODELO DE REGRESION DE PROYECTO TERMINA CON DIFICULTAD 
#Aplicamos los imports primero
# linear algebra
import numpy as np 

#Pandas profile
import pandas_profiling as pp # exploratory data analysis EDA

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

# data visualization
import matplotlib.pyplot as plt # data visualisation
import seaborn as sns #data visualisation
import plotly.express as px #data visualisation
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, norm # Calculo de chi2

# stocks related missing info
import yfinance as yf

# ignoring the warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#ranking the stocks
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import optuna
from wordcloud import WordCloud, STOPWORDS #es para la nube de palabras

# Evaluar si las elimino
import statsmodels.api as sm

#Timer series
import datetime

#!pip install fbprophet --quiet
import plotly.offline as py
py.init_notebook_mode()

#Guardar modelo
import pickle

#Aplicamos los from luego de los imports

# Evaluar si las elimino
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib.pyplot import figure
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

#ranking the stocks
from plotly.subplots import make_subplots

#Prophet Model Stuff
#!pip install fbprophet --quiet

from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.plot import plot_yearly
from prophet.plot import add_changepoints_to_plot
from pickle import FALSE

df_siges = pd.read_csv('src\siges_dataset_proyecto.csv', engine='python')
data=df_siges.drop(['proy_latlng_fk','proy_id_migrado','proy_leccion_aprendida','proy_risk_fk','proy_org_fk','proy_est_pendiente_fk','proy_sol_aceptacion','proy_grp','proy_ult_usuario','proy_ult_mod','proy_ult_origen','proy_version','fecha_cambio_activacion','usuario_cambio_activacion','risk_superado','version','risk_observaciones','ent_codigo','ent_status','ent_horas_estimadas','ent_inicio_es_hito','ent_fin_es_hito','ent_collapsed','ent_assigs','ent_predecesor_dias','version_1','ent_inicio_periodo','ent_fin_periodo','ent_es_referencia','ent_referido','gas_pk','gas_tipo_fk','gas_usu_fk','gas_mon_fk','gas_importe','gas_fecha','gas_obs','gas_aprobado','cal_pk','cal_peso','cal_vca_fk','cal_actualizacion','cal_tipo','cal_ent_fk','cal_prod_fk','cal_tca_fk','latlang_dep_fk','latlang_loc_fk','latlang_codigopostal','latlang_barrio','latlang_loc','media_estado','media_principal','media_orden','media_pub_fecha','media_usr_mod_fk','media_mod_fecha','media_usr_rech_fk','media_rech_fecha','media_contenttype','part_activo','rh_comentario','rh_aprobado','proy_publica_pk','proy_publica_fecha','proy_publica_usu_fk','lecapr_org_fk','lecapr_activo','area_org_fk','area_padre','version_2','area_activo','cro_proy_resources','cro_proy_permiso_escritura','cro_proy_version','est_proy_label','est_proy_version','est_pendiente_pk','est_pendiente_codigo','est_pendiente_nombre','est_pendiente_label','est_pendiente_orden_proceso','est_pendiente_version','org_logo_nombre','org_direccion','org_logo','org_activo','org_token','org_version','org_activo_siges_ind','pre_version','pre_ocultar_pagos_confirmados','fue_org_fk','version_3','version_4','prog_est_pendiente_fk','prog_sol_aceptacion','prog_cro_fk','prog_grp','prog_version','prog_ult_usuario','prog_ult_mod','prog_ult_origen','prog_factor_impacto','progpre_prog_fk','progpre_pre_fk','version_5','fue_prog_pre_org_fk','fue_prog_pre_version','fue_prog_pre_habilitada','cro_prog_pk','cro_prog_ent_seleccionado','cro_prog_ent_borrados','cro_prog_resources','cro_prog_permiso_escritura','cro_prog_permiso_escritura_padre','cro_prog_version','est_prog_lablel','est_prog_version','obj_est_prog_org_fk','proyind_porc_peso_total','proyind_cal_ind','proyind_cal_pend','proyind_avance_par_verde','proyind_avance_fin_verde','proyind_version','proy_otr_cont_fk','proy_otr_ins_eje_fk','proy_otr_ent_fk','proy_otr_origen','proy_otr_plazo','proy_otr_observaciones','proy_otr_cat_fk','proyint_proy_pk','proyint_int_pk','version_7','int_pk','int_rolint_fk','int_observaciones','int_pers_fk','int_orga_fk','int_ent_fk','version_8','int_tipo','int_usuario_fk',],axis=1)
data.shape
indexNames = data[ data['proy_est_fk'] != 5 ].index
# Delete these row indexes from dataFrame
data.drop(indexNames,inplace=True)
indexNames = data[ data['proy_cro_fk'].isnull()].index
# Delete these row indexes from dataFrame
data.drop(indexNames,inplace=True)
data=data.drop(['cro_proy_ent_borrados','mon_proy_cod_pais','mon_pk','mon_nombre','mon_signo','mon_cod_pais','prog_pre_fk',
'pre_pk','pre_base_1','pre_moneda_1','pre_fuente_organi_fk_1','version_6',
'pre_ocultar_pagos_confirmados_1','mon_prog_pre_pk','mon_prog_pre_nombre','mon_prog_pre_signo','mon_prog_pre_cod_pais','mon_prog_pre_version', 
'fue_prog_pre_pk','fue_prog_pre_nombre','proyind_periodo_inicio_ent_fk','proyind_periodo_fin_ent_fk',],axis=1)
data.shape
data=data.drop(['rh_pk','rh_usu_fk','rh_ent_fk','rh_fecha','rh_horas','media_publicable','media_comentario','media_pk','media_tipo_fk','media_link','media_usr_pub_fk','media_pk','media_tipo_fk','media_link','media_usr_pub_fk','media_comentario','media_publicable','latlng_pk','latlng_lat','latlng_lng', ],axis=1)
data.shape 
data=data.drop(['proy_est_fk','part_pk','part_usu_fk','part_ent_fk','part_horas_plan','prog_obj_est_fk','obj_est_prog_pk','obj_est_prog_nombre','obj_est_prog_habilitado' ],axis=1)
data.shape 
data = data.astype({'proy_descripcion': 'string', 'proy_objetivo': 'string','proy_obj_publico': 'string',
'proy_situacion_actual': 'string',
'proy_nombre': 'string',
'proy_factor_impacto':'string',
'risk_nombre':'string',
'risk_efecto':'string',
'risk_estategia':'string',
'risk_disparador':'string',
'risk_contingencia':'string',
'risk_fecha_superado':'string',
'ent_nombre':'string',
'ent_descripcion':'string',
'lecapr_fecha':'string',
'lecapr_desc':'string',
'tipolec_lecc_apre_codigo':'string',
'tipolec_lecc_apre_nombre':'string',
'con_nombre':'string',
'area_nombre':'string',
'area_abreviacion':'string',
'obj_est_proy_nombre':'string',
'obj_est_proy_descripcion':'string',
'mon_proy_nombre':'string',
'mon_proy_signo':'string',
'fue_nombre':'string',
'prog_nombre':'string',
'prog_descripcion':'string',
'prog_objetivo':'string',
'prog_obj_publico':'string',
'prog_usu_cod_adjunto':'string',
'prog_usu_primer_apellido_adjunto':'string',
'prog_usu_primer_nombre_adjunto':'string',
'prog_usu_cod_gerente':'string',
'prog_usu_primer_apellido_gerente':'string',
'prog_usu_primer_nombre_gerente':'string',
'prog_usu_cod_pmofed':'string',
'prog_usu_primer_apellido_pmofed':'string',
'prog_usu_primer_nombre_pmofed':'string',
'prog_usu_cod_sponsor':'string',
'prog_usu_primer_apellido_sponsor':'string',
'prog_usu_primer_nombre_sponsor':'string',
'proy_usu_adjunto_primer_apellido':'string',
'proy_usu_adjunto_primer_nombre':'string',
'proy_usu_adjunto_cod':'string',
'proy_usu_gerente_primer_apellido':'string',
'proy_usu_gerente_primer_nombre':'string',
'proy_usu_gerente_cod':'string',
'proy_usu_pmofed_primer_apellido':'string',
'proy_usu_pmofed_primer_nombre':'string',
'proy_usu_pmofed_cod':'string',
'proy_usu_sponsor_primer_apellido':'string',
'proy_usu_sponsor_primer_nombre' :'string',
'proy_usu_sponsor_cod':'string',})
data.dtypes 
riesgos= data[['proy_pk','risk_pk','risk_nombre','risk_fecha_actu','risk_probabilidad','risk_impacto','risk_ent_fk','risk_fecha_limite','risk_efecto','risk_estategia','risk_disparador','risk_contingencia','risk_fecha_superado','risk_usuario_superado_fk','risk_exposicion']]
riesgos2=riesgos[riesgos['risk_nombre'].str.contains('Desarrollo|pago|Gateway|equipo|soluciones|ws|integracion|proyectos|pasarela|pagos|produccin|dificultades|autoridades|asignacin|disponibilidad|sponsor|retrasos|demoras|autoridades|organismo|proveedor|institucin|institucion|involucramiento|fallas|dificultades|publicacin|personal|cambios|gestion|clave|recursos|contrataciones|compras|adquisiciones|personal|infraestructura', case=False)]
lecciones_app=data[['proy_pk','con_padre_fk','con_org_fk','con_nombre','con_pk','lecaprcon_con_fk','lecaprcon_lecapr_fk','lecapr_pk','lecapr_tipo_fk','lecapr_usr_fk','lecapr_fecha','lecapr_desc','tipolec_lecc_apre_pk','tipolec_lecc_apre_codigo','tipolec_lecc_apre_nombre']]
df_indice_recet1=lecciones_app.reset_index()
lecciones_app1=lecciones_app[lecciones_app['lecapr_desc'].str.contains('Involucrar|personas|organismo|personal|clave|motivado|tiempos|excesivos|implantaciones', case=False)]
lecciones_app2=lecciones_app[lecciones_app['con_nombre'].str.contains('Liderazgo|Humanos|Cronograma|Implantacin|Riesgos|Calidad|Costos|Alcance|Plazos|Comunicaciones|Requerimiento|Recursos', case=False)]
programa=data[['prog_usr_sponsor_fk','prog_usr_pmofed_fk','prog_progindices_fk','prog_nombre','prog_descripcion','prog_objetivo','prog_obj_publico','prog_semaforo_amarillo','prog_semaforo_rojo','prog_activo','prog_fecha_crea','prog_fecha_act','prog_id_migrado','prog_habilitado','est_prog_pk','est_prog_codigo','est_prog_nombre','est_prog_orden_proceso','obj_est_prog_descripcion','prog_usu_id_adjunto','prog_usu_cod_adjunto','prog_usu_primer_apellido_adjunto','prog_usu_primer_nombre_adjunto','prog_usu_id_gerente','prog_usu_cod_gerente','prog_usu_primer_apellido_gerente','prog_usu_primer_nombre_gerente','prog_usu_id_pmofed','prog_usu_cod_pmofed','prog_usu_primer_apellido_pmofed','prog_usu_primer_nombre_pmofed','prog_usu_id_sponsor','prog_usu_cod_sponsor','prog_usu_primer_apellido_sponsor','prog_usu_primer_nombre_sponsor',]]
#Drop irrelevant columns in train data
drop_cols = ['prog_usr_sponsor_fk','prog_usr_pmofed_fk', 'prog_usu_id_adjunto', 'prog_usu_primer_apellido_sponsor','prog_usu_primer_nombre_sponsor','prog_semaforo_amarillo','prog_semaforo_rojo','prog_activo','prog_id_migrado','prog_usu_id_adjunto','prog_usu_id_sponsor','prog_usu_id_gerente','prog_usu_id_pmofed']
programa.drop(drop_cols, axis = 1, inplace = True)
completo=data.drop(['risk_pk','risk_nombre','risk_fecha_actu','risk_probabilidad','risk_impacto','risk_ent_fk','risk_fecha_limite','risk_efecto','risk_estategia','risk_disparador','risk_contingencia','risk_fecha_superado','risk_usuario_superado_fk','risk_exposicion','lecapr_tipo_fk','lecapr_usr_fk','lecapr_fecha','lecapr_desc','tipolec_lecc_apre_pk','tipolec_lecc_apre_codigo','tipolec_lecc_apre_nombre','lecaprcon_con_fk','lecaprcon_lecapr_fk','prog_pk','prog_org_fk','prog_area_fk','prog_est_fk','prog_usr_gerente_fk','prog_usr_adjunto_fk','prog_usr_sponsor_fk','prog_usr_pmofed_fk','prog_progindices_fk','prog_nombre','prog_descripcion','prog_objetivo','prog_obj_publico','prog_semaforo_amarillo','prog_semaforo_rojo','prog_activo','prog_fecha_crea','prog_fecha_act','prog_id_migrado','prog_habilitado','est_prog_pk','est_prog_codigo','est_prog_nombre','est_prog_orden_proceso','obj_est_prog_descripcion','prog_usu_id_adjunto','prog_usu_cod_adjunto','prog_usu_primer_apellido_adjunto','prog_usu_primer_nombre_adjunto','prog_usu_id_gerente','prog_usu_cod_gerente','prog_usu_primer_apellido_gerente','prog_usu_primer_nombre_gerente','prog_usu_id_pmofed','prog_usu_cod_pmofed','prog_usu_primer_apellido_pmofed','prog_usu_primer_nombre_pmofed','prog_usu_id_sponsor','prog_usu_cod_sponsor','prog_usu_primer_apellido_sponsor','prog_usu_primer_nombre_sponsor'],axis=1)
completo.shape
completo=data.drop(['proy_usr_adjunto_fk','proy_usr_sponsor_fk','proy_usr_gerente_fk','proy_usr_pmofed_fk','proy_semaforo_amarillo','proy_semaforo_rojo','proy_activo','proyind_fase_color','proyind_avance_par_azul','proyind_anvance_par_rojo','proyind_avance_fin_azul','proyind_anvance_fin_rojo','proyind_fecha_act','proyind_fecha_act_color','proy_usu_adjunto_id','proy_usu_adjunto_primer_apellido','proy_usu_adjunto_primer_nombre','proy_usu_adjunto_cod','proy_usu_gerente_id','proy_usu_gerente_primer_apellido','proy_usu_gerente_primer_nombre','proy_usu_gerente_cod','proy_usu_pmofed_id','proy_usu_pmofed_primer_apellido','proy_usu_pmofed_primer_nombre','proy_usu_pmofed_cod','proy_usu_sponsor_id','proy_usu_sponsor_primer_apellido','proy_usu_sponsor_primer_nombre','proy_usu_sponsor_cod','proy_peso','risk_pk','risk_nombre','risk_fecha_actu','risk_probabilidad','risk_impacto','risk_ent_fk','risk_fecha_limite','risk_efecto','risk_estategia','risk_disparador','risk_contingencia','risk_fecha_superado','risk_usuario_superado_fk','risk_exposicion','lecapr_pk','lecapr_tipo_fk','lecapr_usr_fk','lecapr_fecha','lecapr_desc','tipolec_lecc_apre_pk','tipolec_lecc_apre_codigo','tipolec_lecc_apre_nombre','lecaprcon_con_fk','lecaprcon_lecapr_fk','prog_pk','prog_org_fk','prog_area_fk','prog_est_fk','prog_usr_gerente_fk','prog_usr_adjunto_fk','prog_usr_sponsor_fk','prog_usr_pmofed_fk','prog_progindices_fk','prog_nombre','prog_descripcion','prog_objetivo','prog_obj_publico','prog_semaforo_amarillo','prog_semaforo_rojo','prog_activo','prog_fecha_crea','prog_fecha_act','prog_id_migrado','prog_habilitado','est_prog_pk','est_prog_codigo','est_prog_nombre','est_prog_orden_proceso','obj_est_prog_descripcion','prog_usu_id_adjunto','prog_usu_cod_adjunto','prog_usu_primer_apellido_adjunto','prog_usu_primer_nombre_adjunto','prog_usu_id_gerente','prog_usu_cod_gerente','prog_usu_primer_apellido_gerente','prog_usu_primer_nombre_gerente','prog_usu_id_pmofed','prog_usu_cod_pmofed','prog_usu_primer_apellido_pmofed','prog_usu_primer_nombre_pmofed','prog_usu_id_sponsor','prog_usu_cod_sponsor','prog_usu_primer_apellido_sponsor','prog_usu_primer_nombre_sponsor'],axis=1)
completo.shape
completo=data.drop(['proy_usr_adjunto_fk','proy_usr_sponsor_fk','proy_usr_gerente_fk','proy_usr_pmofed_fk','proy_semaforo_amarillo','proy_semaforo_rojo','proy_activo','proyind_fase_color','proyind_avance_par_azul','proyind_anvance_par_rojo','proyind_avance_fin_azul','proyind_anvance_fin_rojo','proyind_fecha_act','proyind_fecha_act_color','proy_usu_adjunto_id','proy_usu_adjunto_primer_apellido','proy_usu_adjunto_primer_nombre','proy_usu_adjunto_cod','proy_usu_gerente_id','proy_usu_gerente_primer_apellido','proy_usu_gerente_primer_nombre','proy_usu_gerente_cod','proy_usu_pmofed_id','proy_usu_pmofed_primer_apellido','proy_usu_pmofed_primer_nombre','proy_usu_pmofed_cod','proy_usu_sponsor_id','proy_usu_sponsor_primer_apellido','proy_usu_sponsor_primer_nombre','proy_usu_sponsor_cod','proy_peso','risk_pk','risk_nombre','risk_fecha_actu','risk_probabilidad','risk_impacto','risk_ent_fk','risk_fecha_limite','risk_efecto','risk_estategia','risk_disparador','risk_contingencia','risk_fecha_superado','risk_usuario_superado_fk','risk_exposicion','lecapr_pk','lecapr_tipo_fk','lecapr_usr_fk','lecapr_fecha','lecapr_desc','tipolec_lecc_apre_pk','tipolec_lecc_apre_codigo','tipolec_lecc_apre_nombre','lecaprcon_con_fk','lecaprcon_lecapr_fk','prog_pk','prog_org_fk','prog_area_fk','prog_est_fk','prog_usr_gerente_fk','prog_usr_adjunto_fk','prog_usr_sponsor_fk','prog_usr_pmofed_fk','prog_progindices_fk','prog_nombre','prog_descripcion','prog_objetivo','prog_obj_publico','prog_semaforo_amarillo','prog_semaforo_rojo','prog_activo','prog_fecha_crea','prog_fecha_act','prog_id_migrado','prog_habilitado','est_prog_pk','est_prog_codigo','est_prog_nombre','est_prog_orden_proceso','obj_est_prog_descripcion','prog_usu_id_adjunto','prog_usu_cod_adjunto','prog_usu_primer_apellido_adjunto','prog_usu_primer_nombre_adjunto','prog_usu_id_gerente','prog_usu_cod_gerente','prog_usu_primer_apellido_gerente','prog_usu_primer_nombre_gerente','prog_usu_id_pmofed','prog_usu_cod_pmofed','prog_usu_primer_apellido_pmofed','prog_usu_primer_nombre_pmofed','prog_usu_id_sponsor','prog_usu_cod_sponsor','prog_usu_primer_apellido_sponsor','prog_usu_primer_nombre_sponsor','con_pk','con_nombre','con_org_fk','con_padre_fk','area_director','est_proy_pk','est_proy_codigo','est_proy_nombre','est_proy_orden_prroceso','obj_est_proy_nombre','obj_est_proy_descripcion','obj_est_proy_org_fk','obj_est_proy_habilitado','proy_otr_eta_fk','proy_otr_est_pub_fk','proyind_riesgo_expo','proyind_riesgo_ultact','proyind_riesgo_alto','proyind_metodo_estado','proyind_metodo_sin_aprobar','proyind_periodo_inicio','proyind_periodo_fin','mon_proy_version'],axis=1)
completo.shape
entregables=completo.drop(['org_nombre','proy_fecha_crea','proy_fecha_act','proy_fecha_est_act','proy_fecha_act_pub','proy_area_fk','proy_prog_fk','proy_cro_fk','proy_pre_fk','proy_proyindices_fk','proy_descripcion','proy_objetivo','proy_obj_publico','proy_situacion_actual','proy_nombre','proy_publicable','proy_otr_dat_fk','proy_obj_est_fk','proy_factor_impacto','area_pk','area_nombre','area_abreviacion','area_habilitada','cro_proy_ent_seleccionado','cro_proy_permiso_escritura_padre','pre_base','pre_moneda','pre_fuente_organi_fk','mon_proy_pk','mon_proy_nombre','mon_proy_signo','fue_pk','fue_nombre','fue_habilitada',],axis=1)
entregables.shape
entregebales2=entregables[entregables['ent_nombre'].str.contains('Tramite|Registro|Produccion|agenda|Evolucin|Plan|Taller|Herramienta|Simple|Enseanza|ayuda|Instituciones', case=False)]
cronograma=completo.drop(['org_nombre','proy_fecha_crea','proy_fecha_act','proy_fecha_est_act','proy_fecha_act_pub','proy_area_fk','proy_prog_fk','proy_pre_fk','proy_proyindices_fk','proy_descripcion','proy_objetivo','proy_obj_publico','proy_situacion_actual','proy_nombre','proy_publicable','proy_otr_dat_fk','proy_obj_est_fk','proy_factor_impacto','area_pk','area_nombre','area_abreviacion','area_habilitada','pre_base','pre_moneda','pre_fuente_organi_fk','mon_proy_pk','mon_proy_nombre','mon_proy_signo','fue_pk','fue_nombre','fue_habilitada','ent_pk','ent_cro_fk','ent_id','ent_nombre','ent_nivel','ent_parent','ent_inicio','ent_duracion','ent_fin','ent_coord_usu_fk','ent_esfuerzo','ent_inicio_linea_base','ent_duracion_linea_base','ent_fin_linea_base','ent_predecesor_fk','ent_descripcion','ent_progreso','ent_relevante',],axis=1)
cronograma.shape
presupuesto=completo.drop(['org_nombre','proy_cro_fk','cro_proy_ent_seleccionado','cro_proy_permiso_escritura_padre','proy_fecha_crea','proy_fecha_act','proy_fecha_est_act','proy_fecha_act_pub','proy_prog_fk','proy_proyindices_fk','proy_descripcion','proy_objetivo','proy_obj_publico','proy_situacion_actual','proy_nombre','proy_publicable','proy_otr_dat_fk','proy_obj_est_fk','proy_factor_impacto','ent_pk','ent_cro_fk','ent_id','ent_nombre','ent_nivel','ent_parent','ent_inicio','ent_duracion','ent_fin','ent_coord_usu_fk','ent_esfuerzo','ent_inicio_linea_base','ent_duracion_linea_base','ent_fin_linea_base','ent_predecesor_fk','ent_descripcion','ent_progreso','ent_relevante',],axis=1)
presupuesto.shape
#MODELO DE REGRESION DE DIFICULTAD
df_modelo = pd.read_csv('src\DataModeloRegresion.csv', engine='python')
#Eliminan las filas de esos registros
indexNames = df_modelo[df_modelo['Padre']==0].index
# Delete these row indexes from dataFrame
df_modelo.drop(indexNames,inplace=True)
train, test = train_test_split(df_modelo, test_size = 0.20, shuffle = False)
train.columns[train.isnull().any()]
train['Avance'].isnull().sum()
train['EstadoCronograma'].isnull().sum()
#Eliminan las filas de esos registros
indexNames = train[train['Avance'].isnull()].index
# Delete these row indexes from dataFrame
train.drop(indexNames,inplace=True)
#Eliminan las filas de esos registros
indexNames = train[train['EstadoCronograma'].isnull()].index
# Delete these row indexes from dataFrame
train.drop(indexNames,inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis') # eliminamos eje Y , leyenda de las barras
plt.show()
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
# Make predictions
y_pred = model.predict(X_test)
y_pred
# Check the accuracy score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
accuracy_score(y_test, y_pred)
dificultad_cm = confusion_matrix(y_pred, y_test)
dificultad_cm
print(classification_report(y_pred, y_test))
# Se guarda el modelo de regresion que se pasa a produccion.
filename ='regresionpy.sav'
pickle.dump(model, open(filename, 'wb'))

#PUESTA EN PRODUCCION DE TIME SERIES PESOS

df_pagos = pd.read_csv("src\pagos_moneda_filtro_campos.csv", engine="python", sep=',', quotechar='"', error_bad_lines=False)
#Eliminan las filas de esos registros
indexNames = df_pagos[df_pagos['pag_confirmar'].isnull()].index
# Delete these row indexes from dataFrame
df_pagos.drop(indexNames,inplace=True)
#Eliminan las filas de esos registros de los no confirmados.
indexNames = df_pagos[df_pagos['pag_confirmar']==0].index
# Delete these row indexes from dataFrame
df_pagos.drop(indexNames,inplace=True)
df_indice_recet= df_pagos.reset_index()
import matplotlib.pyplot as plt
plt.scatter(df_pagos['pag_importe_planificado'], df_pagos['pag_importe_real'], color = "#154957", alpha = 0.2)
plt.show()
plt.figure(figsize=(10,8))
d0 = sns.countplot(data=df_indice_recet, x='mon_nombre', color='dodgerblue') 
d2 = sns.displot(data=df_indice_recet, x='pag_importe_real', kde=True, height=8, aspect=1.6, bins=100, binrange=(0, 2100), color='dodgerblue')
d2.set(xlabel='pag_importe_real')
plt.xlim(0, 2100)
plt.show()
#Cambio de los tipo objetos a tipo fecha 
df_pagos['pag_fecha_planificada'] = pd.to_datetime(df_pagos['pag_fecha_planificada'])
df_pagos['pag_fecha_real'] = pd.to_datetime(df_pagos['pag_fecha_real'])
df_pagos.head()
#Eliminan las filas de esos registros
indexNames = df_pagos[df_pagos['mon_pk']==2].index
# Delete these row indexes from dataFrame
df_pagos.drop(indexNames,inplace=True)
pagos_modelo_pesos=df_pagos.drop(['pag_pk','pag_fecha_planificada','pag_importe_planificado','pag_confirmar','mon_pk','mon_nombre'],axis=1)
pagos_modelo_pesos.shape
#Eliminan las filas de esos registros por debaje del 1% del valor medior dado que se suprimen 8% de registros es despeciable.
indexNames = pagos_modelo_pesos[pagos_modelo_pesos['pag_importe_real']<=3019.0].index
# Delete these row indexes from dataFrame
pagos_modelo_pesos.drop(indexNames,inplace=True)
#Eliminamos las filtas por encima de 12 millones dado que son los outlier del boxplot.
indexNames = pagos_modelo_pesos[pagos_modelo_pesos['pag_importe_real']>1200000].index
# Delete these row indexes from dataFrame
pagos_modelo_pesos.drop(indexNames,inplace=True)
pagos_modelo_pesos.sort_values(['pag_fecha_real', 'pag_importe_real'],ascending=False) 
pagos_modelo_pesos
sp = pagos_modelo_pesos.rename(columns={'pag_fecha_real': 'ds','pag_importe_real': 'y'})
sp_sample = sp[(sp.ds.dt.year>2014)]
# Crear la figura y los ejes
fig, ax = plt.subplots()
# Dibujar puntos
ax.scatter(x = sp_sample ['ds'], y = sp_sample ['y'])
plt.ylim(3031.0, 1000000.0)
# Guardar el gráfico en formato png
plt.savefig('diagrama-dispersion.png')
# Mostrar el gráfico
plt.show()
model1 = Prophet(interval_width=0.95)
model1.add_country_holidays(country_name='UY')
model1.fit(sp_sample)
future = model1.make_future_dataframe(periods=30, freq="B")
forecast = model1.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = model1.plot(forecast)
def custom_forecast_plot():
    forecast_length = 30

    prior_df = sp[(sp.ds.dt.year>2014)]
    forecast_df = sp[(sp.ds.dt.year==2021) & (sp.ds.dt.month==1)]
    all_df = pd.concat([prior_df, forecast_df]).sort_values('ds')
    all_df.head()

    all_df_sample = all_df[-forecast_length*5:]
    forecast_sample = forecast[forecast['ds'].isin(all_df["ds"].values)].sort_values('ds')

    prior_vis_df = forecast_sample[-forecast_length*5:-forecast_length]
    forecast_vis_df = forecast_sample[-forecast_length:]

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    ax.plot(all_df_sample['ds'].dt.to_pydatetime(), all_df_sample["y"], '.k')

    ax.plot(prior_vis_df['ds'].dt.to_pydatetime(), prior_vis_df['yhat'], ls='-', c='#0072B2')
    ax.fill_between(prior_vis_df['ds'].dt.to_pydatetime(), prior_vis_df['yhat_lower'], prior_vis_df['yhat_upper'], color='#0072B2', alpha=0.2)

    ax.plot(forecast_vis_df['ds'].dt.to_pydatetime(), forecast_vis_df['yhat'], ls='-', c='#fc7d0b')
    ax.fill_between(forecast_vis_df['ds'].dt.to_pydatetime(), forecast_vis_df['yhat_lower'], forecast_vis_df['yhat_upper'], color='#fc7d0b', alpha=0.2)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)

    ax.set_title('S&P 500 30-Day Forecast')
    plt.show(sns)
    
fig = model1.plot_components(forecast)
fig = model1.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model1, forecast)
final_model_pesos = Prophet(interval_width=0.95, weekly_seasonality=False,seasonality_prior_scale=0.001,changepoint_prior_scale=0.9)
final_model_pesos.add_seasonality(name='yearly', period=365, fourier_order=8)
final_model_pesos.add_country_holidays(country_name='UY')
forecast = final_model_pesos.fit(sp_sample).predict(future)
fig = final_model_pesos.plot(forecast)
# Se guarda el modelo de regresion que se pasa a produccion.
filename ='timeseriesprodpesos.sav'
pickle.dump(model, open(filename, 'wb'))

#PUESTA EN PRODUCCION DE TIME SERIES DOLARES

df_pagos_dolares = pd.read_csv("src\pagos_moneda_filtro_campos.csv", engine="python", sep=',', quotechar='"', error_bad_lines=False)
#Eliminan las filas de esos registros
indexNames = df_pagos_dolares[df_pagos_dolares['pag_confirmar'].isnull()].index
# Delete these row indexes from dataFrame
df_pagos_dolares.drop(indexNames,inplace=True)
#Eliminan las filas de esos registros de los no confirmados.
indexNames = df_pagos_dolares[df_pagos_dolares['pag_confirmar']==0].index
# Delete these row indexes from dataFrame
df_pagos_dolares.drop(indexNames,inplace=True)
#Eliminan las filas de esos registros
indexNames = df_pagos_dolares[df_pagos_dolares['mon_pk']==1 & 3].index
# Delete these row indexes from dataFrame
df_pagos_dolares.drop(indexNames,inplace=True)
df_pagos_dolares=df_pagos_dolares.drop(['pag_pk','pag_fecha_planificada','pag_importe_planificado','pag_confirmar','mon_pk','mon_nombre'],axis=1)
df_pagos_dolares.shape
#Eliminan las filas de esos registros por debaje del 1% del valor medior dado que se suprimen 8% de registros es despeciable.
indexNames = df_pagos_dolares[df_pagos_dolares['pag_importe_real']==0].index
# Delete these row indexes from dataFrame
df_pagos_dolares.drop(indexNames,inplace=True)
#Eliminan las filas de esos registros por debaje del 1% del valor medior dado que se suprimen 8% de registros es despeciable.
indexNames = df_pagos_dolares[df_pagos_dolares['pag_importe_real']>500000].index
# Delete these row indexes from dataFrame
df_pagos_dolares.drop(indexNames,inplace=True)
#Cambio de los tipo objetos a tipo fecha 
df_pagos_dolares['pag_fecha_real'] = pd.to_datetime(df_pagos_dolares['pag_fecha_real'])
df_pagos_dolares.head()
df_pagos_dolares.sort_values(['pag_fecha_real', 'pag_importe_real'],ascending=False) 
df_pagos_dolares
sp = df_pagos_dolares.rename(columns={'pag_fecha_real': 'ds','pag_importe_real': 'y'})
sp_sample1 = sp[(sp.ds.dt.year>2014)]
# Crear la figura y los ejes
fig, ax = plt.subplots()
# Dibujar puntos
ax.scatter(x = sp_sample1 ['ds'], y = sp_sample1 ['y'])
plt.ylim(3031.0, 400000.0)
# Guardar el gráfico en formato png
plt.savefig('diagrama-dispersion.png')
# Mostrar el gráfico
plt.show()
model11 = Prophet(interval_width=0.95)
model11.add_country_holidays(country_name='UY')
model11.fit(sp_sample1)
future = model11.make_future_dataframe(periods=30, freq="B")
forecast = model11.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = model11.plot(forecast)
def custom_forecast_plot():
    forecast_length = 30

    prior_df = sp[(sp.ds.dt.year>2014)]
    forecast_df = sp[(sp.ds.dt.year==2021) & (sp.ds.dt.month==1)]
    all_df = pd.concat([prior_df, forecast_df]).sort_values('ds')
    all_df.head()

    all_df_sample = all_df[-forecast_length*5:]
    forecast_sample = forecast[forecast['ds'].isin(all_df["ds"].values)].sort_values('ds')

    prior_vis_df = forecast_sample[-forecast_length*5:-forecast_length]
    forecast_vis_df = forecast_sample[-forecast_length:]

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    ax.plot(all_df_sample['ds'].dt.to_pydatetime(), all_df_sample["y"], '.k')

    ax.plot(prior_vis_df['ds'].dt.to_pydatetime(), prior_vis_df['yhat'], ls='-', c='#0072B2')
    ax.fill_between(prior_vis_df['ds'].dt.to_pydatetime(), prior_vis_df['yhat_lower'], prior_vis_df['yhat_upper'], color='#0072B2', alpha=0.2)

    ax.plot(forecast_vis_df['ds'].dt.to_pydatetime(), forecast_vis_df['yhat'], ls='-', c='#fc7d0b')
    ax.fill_between(forecast_vis_df['ds'].dt.to_pydatetime(), forecast_vis_df['yhat_lower'], forecast_vis_df['yhat_upper'], color='#fc7d0b', alpha=0.2)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)

    ax.set_title('S&P 500 30-Day Forecast')
    plt.show(sns)
    
fig = model11.plot_components(forecast)
final_model_dolares = Prophet(interval_width=0.95, weekly_seasonality=False,seasonality_prior_scale=0.001,changepoint_prior_scale=0.9)
final_model_dolares .add_seasonality(name='yearly', period=365, fourier_order=8)
final_model_dolares .add_country_holidays(country_name='UY')
forecast = final_model_dolares.fit(sp_sample1).predict(future)
fig = final_model_dolares .plot(forecast)

# Se guarda el modelo de regresion que se pasa a produccion.
filename ='timeseriesprodDOLARESproduccion.sav'
pickle.dump(final_model_dolares, open(filename, 'wb'))

# Heroku uses the last version of python, but it conflicts with 
# some dependencies. Low your version by adding a runtime.txt file
# https://stackoverflow.com/questions/71712258/