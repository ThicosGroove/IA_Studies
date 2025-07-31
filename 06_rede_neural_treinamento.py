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

# Criando rótulos (targets) para classificação
# Vamos classificar baseado no número de filhos (0, 1, 2, 3)
# Convertendo para one-hot encoding
y = np.array([0, 1, 2, 0, 1, 2, 0, 3, 1])  # Rótulos originais
y_one_hot = np.zeros((len(y), 4))  # 4 classes (0, 1, 2, 3 filhos)
y_one_hot[np.arange(len(y)), y] = 1

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.ones((1, n_neurons)) * 0.1

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Funções de ativação
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Função de perda (Loss)
def cross_entropy_loss(y_pred, y_true):
    """
    Calcula a Cross-Entropy Loss
    y_pred: probabilidades preditas pela rede (softmax)
    y_true: rótulos verdadeiros em one-hot encoding
    """
    # Evita log(0) adicionando um pequeno valor
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Cross-entropy: -sum(y_true * log(y_pred))
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss

# Criando as camadas
dim_entrada = X_norm.shape[1]
layer1 = Layer_Dense(dim_entrada, 20)
layer2 = Layer_Dense(20, 4)  # 4 saídas para as 4 classes (0, 1, 2, 3 filhos)

# Forward pass
layer1.forward(X_norm)
layer1_relu = relu(layer1.output)

layer2.forward(layer1_relu)
layer2_softmax = softmax(layer2.output)

# Calculando o loss
loss = cross_entropy_loss(layer2_softmax, y_one_hot)
print(f'Loss inicial: {loss:.4f}')

# Visualização das ativações da primeira camada (ReLU)
plt.figure(figsize=(10, 6))
for i in range(layer1_relu.shape[0]):
    plt.plot(range(layer1_relu.shape[1]), layer1_relu[i], marker='o', label=f'Exemplo {i+1}')
plt.title('Ativação ReLU - Camada 1')
plt.xlabel('Índice do neurônio')
plt.ylabel('Valor de ativação')
plt.grid(True)
plt.legend(loc='upper right', fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# Visualização das probabilidades de saída (Softmax) vs Rótulos verdadeiros
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Probabilidades preditas pela rede
for i in range(layer2_softmax.shape[0]):
    ax1.plot(range(layer2_softmax.shape[1]), layer2_softmax[i], marker='o', label=f'Exemplo {i+1}')
ax1.set_title('Probabilidades Preditas pela Rede (Softmax)')
ax1.set_xlabel('Classe (0, 1, 2, 3 filhos)')
ax1.set_ylabel('Probabilidade')
ax1.grid(True)
ax1.legend(loc='upper right', fontsize=8, ncol=2)

# Gráfico 2: Rótulos verdadeiros
for i in range(y_one_hot.shape[0]):
    ax2.plot(range(y_one_hot.shape[1]), y_one_hot[i], marker='s', label=f'Exemplo {i+1}')
ax2.set_title('Rótulos Verdadeiros (One-Hot)')
ax2.set_xlabel('Classe (0, 1, 2, 3 filhos)')
ax2.set_ylabel('Valor (0 ou 1)')
ax2.grid(True)
ax2.legend(loc='upper right', fontsize=8, ncol=2)

plt.tight_layout()
plt.show()

# Gráfico 3: Comparação lado a lado para cada exemplo
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Comparação: Predições vs Rótulos Verdadeiros', fontsize=16)

for i in range(min(9, len(layer2_softmax))):
    row = i // 3
    col = i % 3
    
    x = range(4)
    axes[row, col].bar([x-0.2 for x in x], layer2_softmax[i], width=0.4, label='Predito', alpha=0.7, color='blue')
    axes[row, col].bar([x+0.2 for x in x], y_one_hot[i], width=0.4, label='Verdadeiro', alpha=0.7, color='red')
    axes[row, col].set_title(f'Exemplo {i+1}')
    axes[row, col].set_xlabel('Classe')
    axes[row, col].set_ylabel('Probabilidade/Valor')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print('Saída da camada 2 (Softmax):')
print(layer2_softmax)
print('\nRótulos verdadeiros (one-hot):')
print(y_one_hot) 

# Visualização simplificada - um exemplo por vez
print("\n=== ANÁLISE DETALHADA DOS EXEMPLOS ===")
for i in range(len(layer2_softmax)):
    print(f"\nExemplo {i+1}:")
    print(f"  Dados: Idade={X[i,0]}, Salário={X[i,1]}, Estudo={X[i,2]}, Sono={X[i,3]}, Filhos={X[i,4]}")
    print(f"  Rótulo verdadeiro: {y[i]} filhos (classe {y[i]})")
    print(f"  Predições da rede:")
    for j in range(4):
        print(f"    Classe {j} ({j} filhos): {layer2_softmax[i,j]:.3f}")
    
    # Gráfico individual para cada exemplo
    plt.figure(figsize=(10, 4))
    
    x = range(4)
    plt.subplot(1, 2, 1)
    plt.bar(x, layer2_softmax[i], color='blue', alpha=0.7, label='Predito pela rede')
    plt.title(f'Exemplo {i+1}: Predições da Rede')
    plt.xlabel('Número de Filhos')
    plt.ylabel('Probabilidade')
    plt.xticks(x, ['0', '1', '2', '3'])
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(x, y_one_hot[i], color='red', alpha=0.7, label='Rótulo verdadeiro')
    plt.title(f'Exemplo {i+1}: Rótulo Verdadeiro')
    plt.xlabel('Número de Filhos')
    plt.ylabel('Valor (0 ou 1)')
    plt.xticks(x, ['0', '1', '2', '3'])
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Gráfico resumo: comparação geral
plt.figure(figsize=(12, 6))
x = np.arange(4)
width = 0.35

# Média das predições da rede
pred_media = np.mean(layer2_softmax, axis=0)
# Média dos rótulos verdadeiros
true_media = np.mean(y_one_hot, axis=0)

plt.bar(x - width/2, pred_media, width, label='Média das Predições', alpha=0.7, color='blue')
plt.bar(x + width/2, true_media, width, label='Média dos Rótulos', alpha=0.7, color='red')

plt.xlabel('Número de Filhos')
plt.ylabel('Probabilidade/Valor')
plt.title('Comparação Geral: Média das Predições vs Rótulos Verdadeiros')
plt.xticks(x, ['0', '1', '2', '3'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nMédia das predições da rede: {pred_media}")
print(f"Média dos rótulos verdadeiros: {true_media}")
print(f"Distribuição real dos filhos: {np.bincount(y)}") 