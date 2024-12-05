# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 08:59:58 2024

@author: edson

Wine dataset 
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Cargamos la base de vinos
df = pd.read_csv("C:/Users/edson/OneDrive/Documents/Septimo semestre/WineQT.csv")
#imprime la parte superior del df
df.head()
#Eliminamos la columna Id
df = df.drop("Id", axis=1)
#Describe para conocer la base
df.describe().T
df["quality"].unique()

#matriz de correlación
correlation_matrix = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap= "coolwarm_r",vmin=-1,vmax=1)
plt.title("Matriz de correlación")
plt.show()

#Correlación con la variable quality
quality_correlation = correlation_matrix ["quality"].sort_values(ascending=False)
print(quality_correlation)

#conocer las columnas
df.columns

#guardar las columnas como objeto
cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol']
for col in cols:
     sns.boxplot(data=df, x='quality', y=col,color="red")
     plt.title(f"Quality vs {col}")
     plt.show()
     
# Gráfico de barras de distribución de quality
ax = sns.countplot(x="quality", data=df)

# Graficar la distribución de la calidad del vino
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="center", fontsize=10, color="black", xytext=(0, 5),
                textcoords="offset points")

plt.title("Distribución de la calidad")
plt.show()

#Se necesita un modelo que favoresca también a los datos que no tienen tanta
# distribución

#se aplica logaritmos para evitar outliers
df_log = df.copy()

for col in cols:
    df_log[col] = np.log1p(df_log[col])
df_log.head()
df.head()

for col in cols:
     sns.boxplot(data=df_log, x='quality', y=col,color="red")
     plt.title(f"Quality vs {col}")
     plt.show()

#modelo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

x = df.drop("quality", axis=1)
y = df["quality"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

print(y_train.value_counts())
print(y_test.value_counts)

model_col = []

for col in cols:
    model_col.append(col)
    model = RandomForestClassifier(random_state=0)
    model.fit(x_train[model_col], y_train)
    y_pred = model.predict(x_test[model_col])
    accuracy = accuracy_score(y_test,y_pred)
    print(f"Precision con {model_col}: {accuracy}")
    
#Train test
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, stratify=y,random_state=42)

print(y_train.value_counts())
print(y_test.value_counts())

for col in cols:
    model_col.append(col)
    model = RandomForestClassifier(random_state=0)
    model.fit(x_train[model_col], y_train)
    y_pred = model.predict(x_test[model_col])
    accuracy = accuracy_score(y_test,y_pred)
    print(f"Precision con {model_col}: {accuracy}")
    
x = df_log.drop("quality", axis=1)
y = df_log["quality"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

model_col = []

for col in cols:
    model_col.append(col)
    model = RandomForestClassifier(random_state=0)
    model.fit(x_train[model_col], y_train)
    y_pred = model.predict(x_test[model_col])
    accuracy = accuracy_score(y_test,y_pred)
    print(f"Precision con {model_col}: {accuracy}")
    
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, stratify=y,random_state=42)

print(y_train.value_counts())
print(y_test.value_counts())


