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
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradientes dos pesos e biases
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradiente para a camada anterior
        self.dinputs = np.dot(dvalues, self.weights.T)

# Funções de ativação
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

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

def cross_entropy_derivative(y_pred, y_true):
    """
    Derivada da Cross-Entropy Loss em relação às predições
    """
    return y_pred - y_true

# Criando as camadas
dim_entrada = X_norm.shape[1]
layer1 = Layer_Dense(dim_entrada, 20)
layer2 = Layer_Dense(20, 4)  # 4 saídas para as 4 classes (0, 1, 2, 3 filhos)

# Parâmetros de treinamento
learning_rate = 0.1
epochs = 1000
losses = []

print("=== TREINAMENTO DA REDE NEURAL ===")
print(f"Learning rate: {learning_rate}")
print(f"Épocas: {epochs}")
print(f"Tamanho do dataset: {len(X_norm)} exemplos")

# Loop de treinamento
for epoch in range(epochs):
    # Forward pass
    layer1.forward(X_norm)
    layer1_output = relu(layer1.output)
    
    layer2.forward(layer1_output)
    layer2_output = softmax(layer2.output)
    
    # Calcular loss
    loss = cross_entropy_loss(layer2_output, y_one_hot)
    losses.append(loss)
    
    # Print progresso a cada 100 épocas
    if epoch % 100 == 0:
        print(f"Época {epoch}: Loss = {loss:.4f}")
    
    # Backward pass (Backpropagation)
    # Gradiente da loss em relação à saída da última camada
    dloss = cross_entropy_derivative(layer2_output, y_one_hot)
    
    # Backpropagation através da camada 2
    layer2.backward(dloss)
    
    # Gradiente através da ativação ReLU da camada 1
    drelu = layer2.dinputs * relu_derivative(layer1.output)
    
    # Backpropagation através da camada 1
    layer1.backward(drelu)
    
    # Atualizar pesos e biases (Gradient Descent)
    layer1.weights -= learning_rate * layer1.dweights
    layer1.biases -= learning_rate * layer1.dbiases
    
    layer2.weights -= learning_rate * layer2.dweights
    layer2.biases -= learning_rate * layer2.dbiases

print(f"\nTreinamento concluído!")
print(f"Loss final: {loss:.4f}")
print(f"Loss inicial: {losses[0]:.4f}")
print(f"Melhoria: {losses[0] - loss:.4f}")

# Gráfico da evolução do loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Evolução do Loss Durante o Treinamento')
plt.xlabel('Época')
plt.ylabel('Loss (Cross-Entropy)')
plt.grid(True)
plt.show()

# Teste final - Forward pass com os pesos treinados
layer1.forward(X_norm)
layer1_output = relu(layer1.output)
layer2.forward(layer1_output)
final_predictions = softmax(layer2.output)

# Análise dos resultados finais
print("\n=== RESULTADOS FINAIS ===")
print("Predições finais da rede (após treinamento):")
for i in range(len(final_predictions)):
    predicted_class = np.argmax(final_predictions[i])
    true_class = y[i]
    correct = "✓" if predicted_class == true_class else "✗"
    print(f"Exemplo {i+1}: Predito={predicted_class}, Verdadeiro={true_class} {correct}")

# Calcular acurácia
predictions = np.argmax(final_predictions, axis=1)
accuracy = np.mean(predictions == y)
print(f"\nAcurácia final: {accuracy:.2%}")

# Visualização final - comparação antes vs depois
plt.figure(figsize=(15, 5))

# Antes do treinamento (usando pesos aleatórios)
np.random.seed(1)
layer1_untrained = Layer_Dense(dim_entrada, 20)
layer2_untrained = Layer_Dense(20, 4)
layer1_untrained.forward(X_norm)
layer1_untrained_output = relu(layer1_untrained.output)
layer2_untrained.forward(layer1_untrained_output)
untrained_predictions = softmax(layer2_untrained.output)

# Gráfico antes do treinamento
plt.subplot(1, 3, 1)
pred_media_antes = np.mean(untrained_predictions, axis=0)
true_media = np.mean(y_one_hot, axis=0)
x = np.arange(4)
width = 0.35
plt.bar(x - width/2, pred_media_antes, width, label='Antes (Predito)', alpha=0.7, color='lightblue')
plt.bar(x + width/2, true_media, width, label='Verdadeiro', alpha=0.7, color='red')
plt.title('Antes do Treinamento')
plt.xlabel('Número de Filhos')
plt.ylabel('Probabilidade')
plt.xticks(x, ['0', '1', '2', '3'])
plt.legend()

# Gráfico depois do treinamento
plt.subplot(1, 3, 2)
pred_media_depois = np.mean(final_predictions, axis=0)
plt.bar(x - width/2, pred_media_depois, width, label='Depois (Predito)', alpha=0.7, color='blue')
plt.bar(x + width/2, true_media, width, label='Verdadeiro', alpha=0.7, color='red')
plt.title('Depois do Treinamento')
plt.xlabel('Número de Filhos')
plt.ylabel('Probabilidade')
plt.xticks(x, ['0', '1', '2', '3'])
plt.legend()

# Gráfico da evolução do loss
plt.subplot(1, 3, 3)
plt.plot(losses)
plt.title('Evolução do Loss')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nComparação:")
print(f"Loss antes: {cross_entropy_loss(untrained_predictions, y_one_hot):.4f}")
print(f"Loss depois: {cross_entropy_loss(final_predictions, y_one_hot):.4f}")
print(f"Melhoria: {cross_entropy_loss(untrained_predictions, y_one_hot) - cross_entropy_loss(final_predictions, y_one_hot):.4f}") 