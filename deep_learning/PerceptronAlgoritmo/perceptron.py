import numpy as np

class Perceptron:
    """
    constructor de la clase Perceptron
    Este método se llama cuando creamos una instancia de la clase Perceptron.
    Recibe tres parámetros: input_size (el número de entradas del perceptrón), 
    learning_rate (la tasa de aprendizaje, por defecto 0.1) 
    y epochs (el número máximo de épocas de entrenamiento, por defecto 100).
    Dentro del constructor, se inicializan los atributos de la neurona: weights (pesos sinápticos), 
    bias (sesgo) y los parámetros de aprendizaje.
    """
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activate(self, x):
        return 1 if x > 0 else 0

    """
    Este método se utiliza para entrenar el perceptrón. 
    Toma un conjunto de datos de entrada inputs y las etiquetas correspondientes labels.
    Dentro del bucle for, para cada entrada y etiqueta, 
    calculamos la predicción actual usando el método predict. 
    Calculamos el error como la diferencia entre la etiqueta real y la predicción.
    Luego, actualizamos los pesos y el sesgo utilizando la regla de aprendizaje del perceptrón. 
    Si el total_error es cero en algún momento, significa que el perceptrón ha convergido y 
    hemos alcanzado una solución. En ese caso, imprimimos un mensaje y salimos del bucle.
    """

    def train(self, inputs, labels):
        for epoch in range(self.epochs):
            total_error = 0
            for input_data, label in zip(inputs, labels):
                prediction = self.predict(input_data)
                error = label - prediction
                total_error += error ** 2

                self.weights += self.learning_rate * error * input_data
                self.bias += self.learning_rate * error

            if total_error == 0:
                print(f"Converged in epoch {epoch + 1}")
                break

    def predict(self, input_data):
        net_input = np.dot(input_data, self.weights) + self.bias
        return self.activate(net_input)

# Ejemplo de uso
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])

perceptron = Perceptron(input_size=2)
perceptron.train(inputs, labels)

# Prueba de predicción
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for data in test_data:
    prediction = perceptron.predict(data)
    print(f"Input: {data}, Prediction: {prediction}")
