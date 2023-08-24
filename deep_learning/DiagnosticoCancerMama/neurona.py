"""
Diagnosticar cancer de mama con una neurona
Conjunto de datos:
 - Tamaño del tumor
 - Textura del tumor
 - Perimetro del tumor
 - Area del tumor
 - Entre otros
 Lo que se busca es identificar si el tumor es maligno o benigno
    benigno = 0
    maligno = 1
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos
breast_cancer = load_breast_cancer()

# X y Y son los datos de entrada y salida
X = breast_cancer.data
Y = breast_cancer.target

#Vizualizar los datos
df = pd.DataFrame(X, columns=breast_cancer.feature_names)
print(df)
print()

#Dividir los datos en entrenamiento y prueba
"""
Aqui X_train y Y_train son los datos de entrenamiento
Donde X_train son los datos de entrada y Y_train son los datos de salida
X_test y Y_test son los datos de prueba
en train_test_split se le pasa los datos de entrada y salida y se usa stratify=Y para que los datos de entrenamiento y prueba tengan la misma proporcion de datos de salida
"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y)
#Imprimir los datos si son malignos o benignos 0 = benigno y 1 = maligno
print(Y)
#Impriir los datos de entrenamiento y prueba
print("Tamaño del conjunto de datos de entrenamiento: ", len(X_train))
print("Tamaño del conjunto de datos de prueba: ", len(X_test))

#Neurona artificial MP
class Neuron:
    def __init__(self):
        self.ordenDeParada = None
    
    def modelo(self, x):
        return (sum(x) >= self.ordenDeParada)
    def prediccion(self, X):
        Y =[]
        for x in X:
            resultado = self.modelo(x)
            Y.append(resultado)

        return np.array(Y)
    
    def fit(self, X, Y):
        accuracy = {}
        #Seleccionamos la orden de parada entre el numero de caracteristicas de los datos de entrada
        for th in range(X.shape[1] + 1):
            self.ordenDeParada = th
            Y_pred = self.prediccion(X)
            accuracy[th] = accuracy_score(Y_pred, Y)
        #Seleccionamos la orden de parada que tenga el mejor accuracy
        self.ordenDeParada = max(accuracy, key=accuracy.get)

"""
import matplotlib.pyplot as plt
print(pd.cut([0.4,2, 4,5,6,0.02,0.06], bins=2, labels=[1,0]))

plt.hist([0.04, 0.3, 4, 5, 6, 0.02, 0.06], bins=2)
plt.show()
print()
"""

# Convertir las matrices de NumPy en DataFrames de Pandas
X_train_df = pd.DataFrame(X_train, columns=breast_cancer.feature_names)
X_test_df = pd.DataFrame(X_test, columns=breast_cancer.feature_names)

# Convertir los datos de entrada a binarios utilizando pd.cut()
X_train_bin = X_train_df.apply(pd.cut, bins=2, labels=[1, 0])
X_test_bin = X_test_df.apply(pd.cut, bins=2, labels=[1, 0])

print(X_train_bin)

# Instanciar la neurona
neurona = Neuron()
# Encontrar el mejor valor de parada
neurona.fit(X_train_bin.to_numpy(), Y_train)
print("Orden de parada: ", neurona.ordenDeParada)
print()

#Realizamos predicciones para ejemplos nuevos que no se encuentran en el conjunto de datos
Y_pred = neurona.prediccion(X_test_bin.to_numpy())
print("Predicciones: ", Y_pred)
print()
#Calculamos la exactitud de las predicciones
print("Exactitud: ", accuracy_score(Y_test, Y_pred))
print()

#Calcular la matriz de confusion
from sklearn.metrics import confusion_matrix
print("Matriz de confusion: \n", confusion_matrix(Y_test, Y_pred))
"""
 [[46  7]
 [11 79]]
 la matriz explica lo siguiente:
 - El 46 equivale a tumores malignos(real)
 - El 7 equivale a que se a equivocado siete veces (falsos positivos)
 - el 11 equivale a que se a equivocado 11 veces en tumores que si son malignos (falsos negativos)
 - el 79 equivale a tumores benignos (real)

 Esta matriz varia

 Exactitud:  0.8111888111888111

 Exactitud: 81%
"""
