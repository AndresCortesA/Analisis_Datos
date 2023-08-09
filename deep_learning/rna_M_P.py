"""
Neurona de McCulloch-Pitts
@autor: Andres Arias 
@fecha: 2023-08-9

Practica 1: Redes Neuronales Artificiales - RNA

Descripción:  esta implementación básica de una neurona de McCulloch-Pitts simula una unidad neuronal simple
que dispara si la suma de los elementos del vector de entrada supera un umbral predeterminado. 
Es importante destacar que esta implementación es muy simplificada
y no representa el funcionamiento de las redes neuronales modernas, pero es un punto de partida histórico en el
desarrollo de conceptos neuronales en la ciencia de las redes neuronales.
"""

# Importar librerias
import numpy as np

class MPNeurona:

    def __init__(self):
        self.parada = None

    def modelo(self, x):
        #Entradas en el recorrido for de abajo: 1+0+1+1 = 3, 1+1+1+1 = 4, 0+0+0+0 = 0, 1+1+1+0 = 3, 0+0+1+1 = 2, 1+0+0+1 = 2
        #No cumple la condicion de parada: 3, 4, 0, 3, 2, 2 - resultado: True, True, False, True, False, False
        return (sum(x) >= self.parada)
    
    def prediccion(self, X):
        #Input X: [1,0,1,1], [1,1,1,1], [0,0,0,0], [1,1,1,0], [0,0,1,1], [1,0,0,1]
        Y = []
        for x in X:
            resultado = self.modelo(x)
            Y.append(resultado)
        return np.array(Y)
    

#Instanciar la clase MPNeurona
neurona = MPNeurona()
#Establecemos una condicion de parada 
neurona.parada = 3
#Hacemos la prediccion con la neurona y diferentes casos de prueba
print(neurona.prediccion([[1,0,1,1], [1,1,1,1], [0,0,0,0], [1,1,1,0], [0,0,1,1], [1,0,0,1]]))
