import tensorflow as tf
import numpy as np
import random
from deap import base, creator, tools

# Załadowanie danych CIFAR-100
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizacja danych


# Definicja modelu CNN
def create_model(params):
    num_conv_layers, filters, kernel_size, dense_units = params

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))

    for _ in range(num_conv_layers):
        model.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(dense_units, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Funkcja fitness oceniająca model
def fitness_function(individual):
    num_conv_layers, filters, kernel_size, dense_units = individual

    # Tworzenie modelu
    model = create_model(individual)

    # Trenowanie modelu przez 1 epokę na próbce danych (dla przyspieszenia)
    history = model.fit(x_train[:10000], y_train[:10000], epochs=1, batch_size=64, verbose=0)

    # Ocena modelu na zbiorze testowym
    _, accuracy = model.evaluate(x_test[:2000], y_test[:2000], verbose=0)
    return accuracy,


# Tworzenie klasy Fitness i indywidualnych rozwiązań
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Inicjalizacja bazy DEAP
toolbox = base.Toolbox()
toolbox.register("num_conv_layers", random.randint, 1, 3)
toolbox.register("filters", random.choice, [32, 64, 128])
toolbox.register("kernel_size", random.choice, [3, 5])
toolbox.register("dense_units", random.choice, [64, 128, 256])
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.num_conv_layers, toolbox.filters, toolbox.kernel_size, toolbox.dense_units), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Definicja operatorów GA
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=256, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness_function)

# Inicjalizacja populacji
population = toolbox.population(n=10)
N_GEN = 5  # Liczba generacji

# Uruchomienie algorytmu genetycznego
for gen in range(N_GEN):
    print(f"Generacja {gen}")

    # Ewaluacja populacji
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Selekcja, krzyżowanie i mutacja
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.5:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Zastąpienie starej populacji nowym potomstwem
    population[:] = offspring

# Wyświetlenie najlepszego rozwiązania
best_individual = tools.selBest(population, 1)[0]
print(f"Najlepsze parametry: {best_individual}")
