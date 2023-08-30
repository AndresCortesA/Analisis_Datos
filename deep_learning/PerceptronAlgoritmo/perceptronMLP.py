import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

# Cargar el conjunto de datos MNIST y dividirlo en entrenamiento y prueba
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los valores de píxeles entre 0 y 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convertir las etiquetas en codificación one-hot
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Crear un modelo secuencial
model = Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Capa de entrada
    tf.keras.layers.Dense(128, activation='relu'),  # Capa oculta con activación ReLU
    tf.keras.layers.Dense(64, activation='relu'),   # Otra capa oculta con activación ReLU
    tf.keras.layers.Dense(10, activation='softmax') # Capa de salida con activación softmax
])

# Compilar el modelo
model.compile(optimizer=SGD(learning_rate=0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy}")
