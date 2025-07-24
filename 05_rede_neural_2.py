import numpy as np
import matplotlib.pyplot as plt
import itertools

np.random.seed(1)

# Exemplo de dados reais: [idade, salário (mil R$), anos de estudo, horas de sono, número de filhos]
X = np.array([
    [25, 3.5, 12, 7, 0],
    [32, 7.2, 16, 6, 1],
    [40, 10.0, 18, 6, 2],
    [22, 2.8, 10, 8, 0],
    [29, 5.0, 14, 7, 1],
    [35, 8.5, 17, 5, 2],
    [28, 4.2, 13, 7, 0],
    [45, 12.0, 20, 6, 3],
    [31, 6.0, 15, 6, 1]
])

# Normalização simples: (x - min) / (max - min) para cada coluna
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.ones((1, n_neurons)) * 0.1

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Funções de ativação
def relu(x):
    return np.maximum(0, x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanh(x):
    return np.tanh(x)

layer1 = Layer_Dense(5, 20)
layer2 = Layer_Dense(20, 10)

layer1.forward(X_norm)
layer1_relu = relu(layer1.output)
layer1_tanh = tanh(layer1.output)
layer1_sigmoid = sigmoid(layer1.output)

# Forward para a segunda camada usando a ativação ReLU da primeira camada
layer2.forward(layer1_relu)
layer2_output = layer2.output

# 1. Matriz de dispersão das features normalizadas (pares de features)
plt.figure(figsize=(10, 8))
features = ['Idade', 'Salário', 'Anos de Estudo', 'Horas de Sono', 'Nº de Filhos']
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for idx, (i, j) in enumerate(itertools.combinations(range(5), 2)):
    label = f'{features[i]} x {features[j]}'
    plt.scatter(X_norm[:, i], X_norm[:, j], c=colors[idx % len(colors)], s=60, edgecolor='k', label=label)
plt.title('Matriz de Dispersão das Features Normalizadas', fontsize=14, pad=20)
plt.xlabel('Feature (normalizada)')
plt.ylabel('Feature (normalizada)')
plt.grid(True)
plt.legend(fontsize=8, loc='best', ncol=2)
plt.tight_layout()
plt.show()

# 2. Ativações de todos os neurônios para cada função de ativação
fig2, axs2 = plt.subplots(3, 1, figsize=(10, 7))
fig2.subplots_adjust(hspace=0.4)

axs2[0].set_title('Ativação ReLU - Todos os neurônios')
for i in range(layer1_relu.shape[0]):
    axs2[0].plot(range(layer1_relu.shape[1]), layer1_relu[i], marker='o', label=f'Exemplo {i+1}')
axs2[0].set_xlabel('Índice do neurônio')
axs2[0].set_ylabel('Valor de ativação')
axs2[0].grid(True)
axs2[0].legend(loc='upper right', fontsize=8, ncol=3)

axs2[1].set_title('Ativação tanh - Todos os neurônios')
for i in range(layer1_tanh.shape[0]):
    axs2[1].plot(range(layer1_tanh.shape[1]), layer1_tanh[i], marker='o', label=f'Exemplo {i+1}')
axs2[1].set_xlabel('Índice do neurônio')
axs2[1].set_ylabel('Valor de ativação')
axs2[1].grid(True)
axs2[1].legend(loc='upper right', fontsize=8, ncol=3)

axs2[2].set_title('Ativação sigmoid - Todos os neurônios')
for i in range(layer1_sigmoid.shape[0]):
    axs2[2].plot(range(layer1_sigmoid.shape[1]), layer1_sigmoid[i], marker='o', label=f'Exemplo {i+1}')
axs2[2].set_xlabel('Índice do neurônio')
axs2[2].set_ylabel('Valor de ativação')
axs2[2].grid(True)
axs2[2].legend(loc='upper right', fontsize=8, ncol=3)

plt.tight_layout()
plt.show()

# 3. Outputs dos neurônios (pares e camada final)
fig3, axs3 = plt.subplots(3, 1, figsize=(10, 8))
fig3.subplots_adjust(hspace=0.4)

axs3[0].set_title('Saída camada 1 (ReLU) - Neurônio 1 vs Neurônio 2')
axs3[0].scatter(layer1_relu[:, 0], layer1_relu[:, 1], c='red', s=80, edgecolor='k')
axs3[0].set_xlabel('Neurônio 1')
axs3[0].set_ylabel('Neurônio 2')
axs3[0].grid(True)

axs3[1].set_title('Saída camada 1 (tanh) - Neurônio 1 vs Neurônio 2')
axs3[1].scatter(layer1_tanh[:, 0], layer1_tanh[:, 1], c='green', s=80, edgecolor='k')
axs3[1].set_xlabel('Neurônio 1')
axs3[1].set_ylabel('Neurônio 2')
axs3[1].grid(True)

axs3[2].set_title('Output da Camada Final (layer2)')
for i in range(layer2_output.shape[0]):
    axs3[2].plot(range(layer2_output.shape[1]), layer2_output[i], marker='o', label=f'Exemplo {i+1}')
axs3[2].set_xlabel('Índice do neurônio (camada 2)')
axs3[2].set_ylabel('Valor de ativação')
axs3[2].grid(True)
axs3[2].legend(loc='upper right', fontsize=8, ncol=2)

plt.tight_layout()
plt.show()

print("Saída da camada 1 (antes da ReLU):")
print(layer1.output)

# Gráfico: Output da camada final (layer2)
plt.figure(figsize=(10, 6))
for i in range(layer2_output.shape[0]):
    plt.plot(range(layer2_output.shape[1]), layer2_output[i], marker='o', label=f'Exemplo {i+1}')
plt.title('Output da Camada Final (layer2)')
plt.xlabel('Índice do neurônio (camada 2)')
plt.ylabel('Valor de ativação')
plt.grid(True)
plt.legend(loc='upper right', fontsize=8, ncol=2)
plt.tight_layout()
plt.show() 