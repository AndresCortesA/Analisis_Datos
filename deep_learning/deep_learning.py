import pandas as pd
import numpy as np


"""
datos principales de pandas
Series: es un array unidimensional que contiene un array de datos y un array de etiquetas, que se denominan índice.

dataframe: es una estructura de datos tabular bidimensional, es decir, los datos se alinean de forma tabular en filas y columnas.

panel: es una estructura de datos tridimensional, es decir, los datos se alinean en un eje de filas, un eje de columnas y un eje de matrices.
"""

#Intanciar un objeto de la clase Series

S = pd.Series([1,2,3,4,5,6,7,8,9,10])
print(S)
print()

altura  = {"Pedro": 180, 
           "Juan": 170, 
           "Maria": 160, 
           "Jose": 150}
s = pd.Series(altura)
print(s)
print()

#inicializacion con algunos indices de la serie
s = pd.Series(altura, index = ["Pedro", "Juan", "Maria"])
print(s)
print()

#Iniciacion con un escalar
s = pd.Series(30, ["test1", "test2", "test3"])
print(s)
print()

#_--------------------------------------------------------------_

s1 = pd.Series([1,2,3,4], index = ["a", "b", "c", "d"])
print(s1)
print(np.sum(s1))
print()


#_--------------------------------------------------------------_ Represantacion de datos de forma grafica

temperaturas = [4.1, 5.5, 6.7, 8.9, 10.1, 11.2, 12.3, 11.4, 9.5, 8.6, 7.7, 6.8, 0.1]

s = pd.Series(temperaturas, name="Temperaturas")
print(s)
print()

"""
0      4.1
1      5.5
2      6.7
3      8.9
4     10.1
5     11.2
6     12.3
7     11.4
8      9.5
9      8.6
10     7.7
11     6.8
12     0.1
Name: Temperaturas, dtype: float64
"""

#_--------------------------------------------------------------_ Graficar datos
import matplotlib.pyplot as plt
s.plot()
#plt.show()

#_--------------------------------------------------------------_ Graficar datos con un estilo dataframe
personas = {
    "Peso": pd.Series([68, 83, 112], index=["Pedro", "Juan", "Maria"]),
    "Altura": pd.Series({"Pedro": 180, "Juan": 170, "Maria": 160}),
    "Hijos": pd.Series([2, 3], index=["Maria", "Juan"])
}

df = pd.DataFrame(personas)
print(df)
print()

df = pd.DataFrame(personas, columns=["Altura", "Peso"], index=["Maria", "Pedro"])
print(df)
print()

"""

       Peso  Altura  Hijos
Juan     83     170    3.0
Maria   112     160    2.0
Pedro    68     180    NaN

       Altura  Peso
Pedro     180    68
Maria     160   112

"""

"""
Creacion de un dataframe de a partir de una lista de listas
Expecificar columnas y indices por aparte
"""

valores = [
       [189, 2, 90],
       [140, 1, 63],
       [176, 2, 90]
]

df = pd.DataFrame(valores, columns = ["Altura", "Hijos", "Peso"], index = ["Juan", "Maria", "Andres"])
print(df)
print()

# Creacion de un dataframe de a partir de un diccionario de listas
datos = {
    #Columnas            #Indices se corresponden con los datos que estan dentro de los valores 
    "Altura": {"Juan": 189, "Maria": 140, "Andres": 176},
    "Peso": {"Juan": 90, "Maria": 63, "Andres": 90},
}

df = pd.DataFrame(datos)
print(df)
"""
        Altura  Hijos  Peso
Juan       189      2    90
Maria      140      1    63
Andres     176      2    90
"""
print()
personas2 = {
    "Sueldo": pd.Series([1000,2000,3000], ["Andres", "Juan", "Santiago"]),
    "Puesto": pd.Series(["Gestor de proyectos", "Contador privado", "Desarrollador web"], ["Andres", "Juan", "Santiago"]),
    "Anios en la empresa": pd.Series([1,2,3], ["Andres", "Juan", "Santiago"])
}

df = pd.DataFrame(personas2)
print(df)
print()
#_--------------------------------------------------------------_ Acceder a los datos de un dataframe
print(df["Sueldo"])
print()
print(df[["Sueldo", "Puesto"]])
print()
print(df[df["Sueldo"] > 1500])
print()

#_--------------------------------------------------------------_ Añadir una columna a un dataframe
df["Cumpleanios"] = [2002, 1998, 1996]
print(df)
print()
#_--------------------------------------------------------------_ Añadir una nueva columna calculada a un dataframe
df["Sueldo anual"] = df["Sueldo"] * 12 
df["Anios"] = 2023 - df["Cumpleanios"]
print(df)
print()
#_--------------------------------------------------------------_ Añadir una nueva columna creando un dataframe nuevo
df_mod = df.assign(mascotas = [0, 1, 5])
print(df_mod)
print()
#_--------------------------------------------------------------_ Eliminar una columna de un dataframe (modifica el dataframe original)
del df["Anios"]
print(df)
print()
#Eliminar una columna exsitente devolviendo una copia del dataframe resultante (no modifica el dataframe original)
df_mod = df.drop("Sueldo anual", axis = 1)
print(df_mod)
print()