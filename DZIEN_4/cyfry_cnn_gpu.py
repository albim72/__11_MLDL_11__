import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

#sprawdź czy GPU jest dostępne
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU jest dostepne {gpus}")
else:
    print("GPU niedostępne!")


# Załaduj dane MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Przekształć dane wejściowe do odpowiedniego formatu: (liczba_próbek, wysokość, szerokość, liczba_kanałów)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Normalizuj dane wejściowe, aby wartości mieściły się w zakresie [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Zakoduj etykiety jako one-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Zbuduj model CNN
model = models.Sequential()

# Pierwsza warstwa konwolucyjna (32 filtry, 3x3, ReLU)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Druga warstwa konwolucyjna (64 filtry, 3x3, ReLU)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Trzecia warstwa konwolucyjna (64 filtry, 3x3, ReLU)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Płaska warstwa, aby przekazać dane do warstwy w pełni połączonej
model.add(layers.Flatten())

# Warstwa w pełni połączona (64 neurony)
model.add(layers.Dense(64, activation='relu'))

# Warstwa wyjściowa (10 klas, softmax)
model.add(layers.Dense(10, activation='softmax'))

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Podsumowanie modelu
model.summary()

# Trening modelu
with tf.device('/GPU:0'):
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Ocena modelu
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
