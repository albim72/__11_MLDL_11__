import numpy as np
from simplenn import SimpleNeuralNetwork

network = SimpleNeuralNetwork()
print(network)

#dane wejściowe
train_inputs = np.array([[1,1,0],[1,1,1],[1,1,0],[1,0,0],[0,1,1],[0,1,0]])
train_outpts = np.array([[1,0,1,1,0,1]]).T
trai_iterators = 50_000

#trening sieci neuronowej
network.train(train_inputs,train_outpts,trai_iterators)
print(f"\nwagi po treningu:\n{network.weights}")

#dane testowe
testdata = np.array([[1,1,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]])

print("predykcja z użyciem modelu")
for data in testdata:
    print(f"wynik dla {data} wynosi: {network.propagation(data)}")
