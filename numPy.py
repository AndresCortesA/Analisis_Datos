import numpy as np
import matplotlib.pyplot as plt

a = np.array([(1, 2, 3), (4, 5, 6)])
b = np.array([1, 2, 3])

"""
Autor: github.com/AndresCortesA
Fecha: 2023-07-18
Descripcion: Este programa es para aprender a usar la libreria de numpy para empezar análisis de datos

Nombre: Andres Cortes
Practica: 1
"""

print("Matriz a 2D")
print(a)
print()
print("Matriz b 1D")
print(b)
print()

print("Raiz cuadrada de b")
print(np.sqrt(b))
print("Exponencial de b")
print(np.exp(b))
print()

# Esto es para ver las dimensiones de los arreglos
print("Dimensiones de los arreglos")
print(a.shape)  # en este caso es una matriz de 2x3 lo que devuelve (2,3)
print(b.shape)
print()

# en este caso vamos a dividir las matrices
print("Division de matrices")
g = np.divide(a, b)
print(g)
print()

# Ahora vamos a multiplicar las matrices
print("Multiplicacion de matrices")
h = np.multiply(a, b)
print(h)

# Producto punto con visualización de matrices gráficamente
print("Producto punto con vectores graficamente")
vector1 = np.array([2, 3])
vector2 = np.array([3, 4])

# Producto punto
vProductoPunto = np.dot(vector1, vector2)
print("Producto punto con dot:", vProductoPunto)

# Visualización de vectores gráficamente
plt.plot(vector1, vector2, label="Producto punto")
plt.xlabel("Vector 1", color="red")
plt.ylabel("Vector 2", color="blue", labelpad=10)
plt.title("Producto Punto de Vectores")
plt.legend()
plt.grid(True)
plt.show()
