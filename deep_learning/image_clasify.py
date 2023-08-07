#Conjunto de datos de moda de MNIST
from tensorflow.keras import datasets
fashion_mnist = datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#Visualizacion de conjunto de datos
import numpy as np
import matplotlib.pyplot as plt

#Define figure para mostrar imagenes
plt.figure(figsize=(20,4))

#Definimos el for para mostrar las imagenes, el zip es para unir los datos de X_train y y_train 
for index, img in zip(range(1, 9), X_train[:8]):
    plt.subplot(1, 8, index)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.title("Ejemplo: " + str(index))

#Mostramos las imagenes
plt.show()

print("Forma de los datos de entrenamiento: ", len(X_train))
print("Forma de los datos de prueba: ", len(X_test))