import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# wczytywanie argumentów z lini komend
parser = argparse.ArgumentParser(description='MLPRegressor')
parser.add_argument('--hidden_neurons', type=int, default=10, help='Liczba neuronów w warstwie ukrytej')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Współczynnik uczenia')
parser.add_argument('--epochs', type=int, default=10000, help='Liczba epok')
parser.add_argument('--outdir', type=str, default='results', help='Katalog wyjściowy')
args = parser.parse_args()

num_hidden_neurons = args.hidden_neurons
learning_rate = args.learning_rate
num_epochs = args.epochs
outdir = args.outdir
os.makedirs(outdir, exist_ok=True)

basic_function = lambda x1, x2: np.cos(x1 * x2) * np.cos(2 * x1)

# generowanie danych

num_train_samples = 1000
x1_train = np.random.uniform(0, np.pi, num_train_samples)
x2_train = np.random.uniform(0, np.pi, num_train_samples)
x_train = np.column_stack((x1_train, x2_train))
y_train = basic_function(x1_train, x2_train)

# wykres scatter
plt.scatter(x1_train, x2_train, c=y_train, cmap='viridis', s=10)
plt.xlabel('X1_train')
plt.ylabel('X2_train')
plt.title('Zbiór próbek: y = cos(X1_train * X2_train * cos(2 * X1_train))')
plt.colorbar(label='y_train')
plt.savefig(f'{outdir}/scatter_samples.png')
plt.close()


# wykres powierzchni (surface)
grid = 50
x1grid = np.linspace(0, np.pi, grid)
x2grid = np.linspace(0, np.pi, grid)
x1_mesh, x2_mesh = np.meshgrid(x1grid, x2grid)
surface_values = basic_function(x1_mesh, x2_mesh)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_mesh, x2_mesh, surface_values, cmap='viridis', alpha=0.8)
ax.set_xlabel('X1_train')
ax.set_ylabel('X2_train')
ax.set_zlabel('y_true')
ax.set_title('Powierzchnia funkcji: y = cos(X1_train * X2_train * cos(2 * X1_train))')
plt.savefig(f'{outdir}/surface_true.png')
plt.close()


# wykres powierzchni + punkty treningowe (surface + scatter)
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_mesh, x2_mesh, surface_values, cmap='viridis', alpha=0.8)
ax.scatter(x1_train, x2_train, y_train, c='b', s=10, label='Punkty treningowe')
ax.set_xlabel('X1_train')
ax.set_ylabel('X2_train')
ax.set_zlabel('y')
ax.set_title('Powierzchnia funkcji + punkty treningowe')
ax.legend()
plt.savefig(f'{outdir}/surface_with_samples.png')
plt.close()


# model MLP
class MLPModel:
    def __init__(self, hidden_neurons, learning_rate, epochs):
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_weights = np.random.randn(hidden_neurons,3) * 0.1
        self.output_weights = np.random.randn(hidden_neurons + 1) *0.1
        
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x_vector):
       biased_input = np.concatenate(([1.0], x_vector))                                  # bias + wejscia
       pre_hidden = self.hidden_weights.dot(biased_input)                                # wejscie do warstwy ukrytej
       post_hidden = self.sigmoid(pre_hidden)                                            # aktywacja warstwy ukrytej (sigmoid)
       biased_hidden = np.concatenate(([1.0], post_hidden))                              # bias + output warstwy ukrytej
       network_output = self.output_weights.dot(biased_hidden)                           # wyjscie z sieci
       
       return biased_input, pre_hidden, post_hidden, biased_hidden, network_output
   
    def train (self, x_train, y_train):
        num_samples = x_train.shape[0]
        for _ in range(self.epochs):
            idx = np.random.randint(num_samples)
            x_vector = x_train[idx]
            y_value = y_train[idx]
            
            _, pre_hidden, post_hidden, biased_hidden, y_pred = self.forward(x_vector)
            
            error = y_pred - y_value
            
            # aktualizacja wag wejściowych
            # gradient dla każdej wagi
            gradient_output = error * biased_hidden
            self.output_weights -= self.learning_rate * gradient_output
            
            # aktualizacja wag ukrytych
            # dla każdego neuronu ukrytego k
            for k in range(self.hidden_neurons):
                # pochodna sigmiody w punkcie post_hidden[k]
                sigmoid_derivative = post_hidden[k] * (1.0 - post_hidden[k])
                # lokalny błąd dla neuronu k
                dleta_k = error * self.output_weights[k + 1] * sigmoid_derivative
                
                # aktualizacja wag dla x1 i x2
                self.hidden_weights[k, 1:] -= self.learning_rate * dleta_k * x_vector
                # aktualizacja wagi biasu
                self.hidden_weights[k, 0] -= self.learning_rate * dleta_k
                
    def predict(self, x_matrix):
        predictions = []
        for x_vector in x_matrix:
            y_pred = self.forward(x_vector)[-1]
            predictions.append(y_pred)
        return np.array(predictions)
    
    
# ======================================================================
# trenowanie modelu i zapis wag
mlp = MLPModel(num_hidden_neurons, learning_rate, num_epochs)
mlp.train(x_train, y_train)

np.savetxt(f'{outdir}/hidden_weights.txt', mlp.hidden_weights, fmt = '%.5f', header = 'hidden_weights rows: [bias, w_x1, w_x2]' )
np.savetxt(f'{outdir}/output_weights.txt', mlp.output_weights[np.newaxis,:], fmt = '%.5f', header = 'output_weights: [bias, w_h1, w_h2, ...]' )


# ewaluacja na zbiorze testowym
num_test_samples = 10000
x1_test = np.random.uniform(0, np.pi, num_test_samples)
x2_test = np.random.uniform(0, np.pi, num_test_samples)
x_test = np.column_stack((x1_test, x2_test))
y_test = basic_function(x1_test, x2_test)
y_pred_custom = mlp.predict(x_test)
mae_custom = mean_absolute_error(y_test, y_pred_custom)
print(f'MAE dla własnej MLP: {mae_custom:.4f}')

sklearn_mlp = MLPRegressor(hidden_layer_sizes=(num_hidden_neurons,), activation='logistic', solver='sgd', learning_rate_init=learning_rate, max_iter=num_epochs)
sklearn_mlp.fit(x_train, y_train)
y_pred_sklearn = sklearn_mlp.predict(x_test)
mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)
print(f'MAE dla sklearn MLPRegressor: {mae_sklearn:.4f}')


# surface plot aproksymacji MLP
grid_points = np.column_stack((x1_mesh.ravel(), x2_mesh.ravel()))
predicted_surface = mlp.predict(grid_points).reshape(x1_mesh.shape)
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_mesh, x2_mesh, predicted_surface, cmap='viridis', alpha=0.8)
ax.set_xlabel('X1_test')
ax.set_ylabel('X2_test')
ax.set_zlabel('y_pred')
ax.set_title('Powierzchnia aproksymacji przez MLP')
plt.savefig(f'{outdir}/surface_predicted.png')
plt.close()


#====================================================================
# eksperymenty: wpływ neuronóe, epok, learning rate (współczynnika uczenia)

# wpływ liczby neuronów
neuron_list = [5, 10, 50, 100]
mae_neurons = []
for n in neuron_list:
    model = MLPModel(n, learning_rate, num_epochs) 
    model.train(x_train, y_train)
    mae_neurons.append(mean_absolute_error(y_test, model.predict(x_test)))
    
plt.figure(figsize=(10, 5))
plt.plot(neuron_list, mae_neurons, marker='o')
plt.xlabel('Liczba neuronów')
plt.ylabel('MAE')
plt.title('Wpływ liczby neuronów na MAE')
plt.savefig(f'{outdir}/neurons_mae.png')
plt.close()


# wpływ liczby epok
epoch_list = [100, 2000, 5000, 10000]
mae_epochs = []
for e in epoch_list:
    model = MLPModel(num_hidden_neurons, learning_rate, e) 
    model.train(x_train, y_train)
    mae_epochs.append(mean_absolute_error(y_test, model.predict(x_test)))
    
plt.figure(figsize=(10, 5))
plt.plot(epoch_list, mae_epochs, marker='o')
plt.xlabel('Liczba epok')
plt.ylabel('MAE')
plt.title('Wpływ liczby epok na MAE')
plt.savefig(f'{outdir}/epochs_mae.png')
plt.close()


# wpływ współczynnika uczenia
learning_rate_list = [0.0001, 0.001, 0.01, 0.1]
mae_learning_rate = []
for lr in learning_rate_list:
    model = MLPModel(num_hidden_neurons, lr, num_epochs) 
    model.train(x_train, y_train)
    mae_learning_rate.append(mean_absolute_error(y_test, model.predict(x_test)))
    
plt.figure(figsize=(10, 5))
plt.plot(learning_rate_list, mae_learning_rate, marker='o')
plt.xscale('log')
plt.xlabel('Współczynnik uczenia')
plt.ylabel('MAE')
plt.title('Wpływ współczynnika uczenia na MAE')
plt.savefig(f'{outdir}/learning_rate_mae.png')
plt.close()