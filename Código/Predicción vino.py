# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:10:34 2024

@author: edson

WineQ tree
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv("C:/Users/edson/OneDrive/Documents/Septimo semestre/WineQT.csv")
df = df.drop("Id", axis=1)

# Definir X e y
col = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
X = df[col]
y = df["quality"]

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo de clasificación
model = DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Precisión del modelo:", accuracy)
print("Matriz de confusión:\n", conf_matrix)
print("Reporte de clasificación:\n", report)

# Graficar el árbol de decisión
plt.figure(figsize=(20,10))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=[str(x) for x in np.unique(y)], rounded=True)
plt.show()



import pandas as pd
import numpy as np

# Supongamos que ya tienes el modelo entrenado (como en el código anterior)
# Define las características del nuevo vino
nuevo_vino = {
    'fixed acidity': 7.4,
    'volatile acidity': 0.7,
    'citric acid': 0.0,
    'residual sugar': 1.9,
    'chlorides': 0.076,
    'free sulfur dioxide': 11.0,
    'total sulfur dioxide': 34.0,
    'density': 0.9978,
    'pH': 3.51,
    'sulphates': 0.56,
    'alcohol': 9.4
}

# Convertir a DataFrame
nuevo_vino_df = pd.DataFrame([nuevo_vino])

# Predecir la calidad
prediccion_calidad = model.predict(nuevo_vino_df)
print("La calidad predicha para el nuevo vino es:", prediccion_calidad[0])
