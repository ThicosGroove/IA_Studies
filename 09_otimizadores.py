import numpy as np
import matplotlib.pyplot as plt

print("=== ESTUDO DOS OTIMIZADORES ===")
print("Implementando e comparando diferentes otimizadores do zero\n")

# ========== CONFIGURAÇÃO INICIAL ==========
np.random.seed(42)

# Dados de exemplo
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

# Normalização
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Rótulos
y = np.array([0, 1, 2, 0, 1, 2, 0, 3, 1])
y_one_hot = np.zeros((len(y), 4))
y_one_hot[np.arange(len(y)), y] = 1

# Classe Layer_Dense
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
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# Funções de ativação
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Funções de loss
def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss

def cross_entropy_derivative(y_pred, y_true):
    return y_pred - y_true

# ========== IMPLEMENTAÇÃO DOS OTIMIZADORES ==========

class SGD:
    """Stochastic Gradient Descent básico"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, weights, biases, dweights, dbiases):
        weights -= self.learning_rate * dweights
        biases -= self.learning_rate * dbiases
        return weights, biases

class SGD_Momentum:
    """SGD com Momentum"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v_weights = None
        self.v_biases = None
    
    def update(self, weights, biases, dweights, dbiases):
        # Inicializar velocidades se necessário
        if self.v_weights is None:
            self.v_weights = np.zeros_like(weights)
            self.v_biases = np.zeros_like(biases)
        
        # Atualizar velocidades
        self.v_weights = self.momentum * self.v_weights - self.learning_rate * dweights
        self.v_biases = self.momentum * self.v_biases - self.learning_rate * dbiases
        
        # Atualizar parâmetros
        weights += self.v_weights
        biases += self.v_biases
        
        return weights, biases

class RMSprop:
    """RMSprop - Root Mean Square Propagation"""
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.v_weights = None
        self.v_biases = None
    
    def update(self, weights, biases, dweights, dbiases):
        # Inicializar velocidades se necessário
        if self.v_weights is None:
            self.v_weights = np.zeros_like(weights)
            self.v_biases = np.zeros_like(biases)
        
        # Atualizar velocidades (média móvel dos gradientes ao quadrado)
        self.v_weights = self.rho * self.v_weights + (1 - self.rho) * (dweights ** 2)
        self.v_biases = self.rho * self.v_biases + (1 - self.rho) * (dbiases ** 2)
        
        # Atualizar parâmetros
        weights -= self.learning_rate * dweights / (np.sqrt(self.v_weights) + self.epsilon)
        biases -= self.learning_rate * dbiases / (np.sqrt(self.v_biases) + self.epsilon)
        
        return weights, biases

class Adam:
    """Adam - Adaptive Moment Estimation"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = None
        self.m_biases = None
        self.v_weights = None
        self.v_biases = None
        self.t = 0
    
    def update(self, weights, biases, dweights, dbiases):
        # Inicializar momentos se necessário
        if self.m_weights is None:
            self.m_weights = np.zeros_like(weights)
            self.m_biases = np.zeros_like(biases)
            self.v_weights = np.zeros_like(weights)
            self.v_biases = np.zeros_like(biases)
        
        self.t += 1
        
        # Atualizar momentos (média móvel dos gradientes)
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * dweights
        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * dbiases
        
        # Atualizar velocidades (média móvel dos gradientes ao quadrado)
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (dweights ** 2)
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (dbiases ** 2)
        
        # Correção de bias
        m_weights_corrected = self.m_weights / (1 - self.beta1 ** self.t)
        m_biases_corrected = self.m_biases / (1 - self.beta1 ** self.t)
        v_weights_corrected = self.v_weights / (1 - self.beta2 ** self.t)
        v_biases_corrected = self.v_biases / (1 - self.beta2 ** self.t)
        
        # Atualizar parâmetros
        weights -= self.learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + self.epsilon)
        biases -= self.learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected) + self.epsilon)
        
        return weights, biases

class AdamW:
    """AdamW - Adam com Weight Decay corrigido"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m_weights = None
        self.m_biases = None
        self.v_weights = None
        self.v_biases = None
        self.t = 0
    
    def update(self, weights, biases, dweights, dbiases):
        # Inicializar momentos se necessário
        if self.m_weights is None:
            self.m_weights = np.zeros_like(weights)
            self.m_biases = np.zeros_like(biases)
            self.v_weights = np.zeros_like(weights)
            self.v_biases = np.zeros_like(biases)
        
        self.t += 1
        
        # Adicionar weight decay aos gradientes
        dweights += self.weight_decay * weights
        
        # Atualizar momentos
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * dweights
        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * dbiases
        
        # Atualizar velocidades
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (dweights ** 2)
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (dbiases ** 2)
        
        # Correção de bias
        m_weights_corrected = self.m_weights / (1 - self.beta1 ** self.t)
        m_biases_corrected = self.m_biases / (1 - self.beta1 ** self.t)
        v_weights_corrected = self.v_weights / (1 - self.beta2 ** self.t)
        v_biases_corrected = self.v_biases / (1 - self.beta2 ** self.t)
        
        # Atualizar parâmetros
        weights -= self.learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + self.epsilon)
        biases -= self.learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected) + self.epsilon)
        
        return weights, biases

# ========== FUNÇÃO DE TREINAMENTO COM OTIMIZADORES ==========

def train_with_optimizer(X, y_true, optimizer_class, optimizer_params, epochs=1000):
    """Treina uma rede neural usando um otimizador específico"""
    np.random.seed(42)  # Mesma seed para comparação justa
    
    # Criar rede neural
    layer1 = Layer_Dense(X.shape[1], 20)
    layer2 = Layer_Dense(20, 4)
    
    # Criar otimizadores separados para cada camada
    optimizer1 = optimizer_class(**optimizer_params)
    optimizer2 = optimizer_class(**optimizer_params)
    
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        layer1.forward(X)
        layer1_output = relu(layer1.output)
        
        layer2.forward(layer1_output)
        layer2_output = softmax(layer2.output)
        
        # Loss
        loss = cross_entropy_loss(layer2_output, y_true)
        losses.append(loss)
        
        # Backward pass
        dloss = cross_entropy_derivative(layer2_output, y_true)
        
        layer2.backward(dloss)
        drelu = layer2.dinputs * relu_derivative(layer1.output)
        layer1.backward(drelu)
        
        # Atualizar pesos usando otimizadores separados para cada camada
        layer1.weights, layer1.biases = optimizer1.update(
            layer1.weights, layer1.biases, layer1.dweights, layer1.dbiases
        )
        layer2.weights, layer2.biases = optimizer2.update(
            layer2.weights, layer2.biases, layer2.dweights, layer2.dbiases
        )
    
    # Forward pass final para obter predições
    layer1.forward(X)
    layer1_output = relu(layer1.output)
    layer2.forward(layer1_output)
    final_output = softmax(layer2.output)
    
    return losses, final_output

# ========== COMPARAÇÃO DOS OTIMIZADORES ==========

print("=== COMPARAÇÃO DOS OTIMIZADORES ===")

# Configurar otimizadores com suas classes e parâmetros
optimizer_configs = {
    'SGD': (SGD, {'learning_rate': 0.1}),
    'SGD + Momentum': (SGD_Momentum, {'learning_rate': 0.1, 'momentum': 0.9}),
    'RMSprop': (RMSprop, {'learning_rate': 0.001, 'rho': 0.9}),
    'Adam': (Adam, {'learning_rate': 0.001, 'beta1': 0.9, 'beta2': 0.999}),
    'AdamW': (AdamW, {'learning_rate': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'weight_decay': 0.01})
}

# Treinar com cada otimizador
results = {}
for name, (optimizer_class, params) in optimizer_configs.items():
    print(f"\nTreinando com {name}...")
    losses, predictions = train_with_optimizer(X_norm, y_one_hot, optimizer_class, params, epochs=1000)
    
    # Calcular acurácia
    pred_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(pred_classes == y)
    
    results[name] = {
        'losses': losses,
        'predictions': predictions,
        'final_loss': losses[-1],
        'initial_loss': losses[0],
        'improvement': losses[0] - losses[-1],
        'accuracy': accuracy
    }
    
    print(f"  Loss inicial: {losses[0]:.4f}")
    print(f"  Loss final: {losses[-1]:.4f}")
    print(f"  Melhoria: {losses[0] - losses[-1]:.4f}")
    print(f"  Acurácia: {accuracy:.2%}")

# ========== VISUALIZAÇÕES COMPARATIVAS ==========

print("\n=== CRIANDO VISUALIZAÇÕES COMPARATIVAS ===")

# Configurar cores
colors = ['#FF5722', '#2196F3', '#4CAF50', '#9C27B0', '#FF9800']

# Figura 1: Evolução do Loss
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
ax1.set_title('Comparação da Evolução do Loss por Otimizador', fontsize=16, fontweight='bold', pad=20)

for i, (name, result) in enumerate(results.items()):
    ax1.plot(result['losses'], label=name, color=colors[i], linewidth=2, alpha=0.8)

ax1.set_xlabel('Época', fontsize=12)
ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Figura 2: Análise Detalhada
fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2, figsize=(15, 12))
fig2.suptitle('Análise Detalhada dos Otimizadores', fontsize=18, fontweight='bold')

# 2a. Loss final comparativo
loss_finals = [result['final_loss'] for result in results.values()]
optimizer_names = list(results.keys())

bars_final = ax2a.bar(optimizer_names, loss_finals, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax2a.set_title('Loss Final Comparativo', fontweight='bold', fontsize=14)
ax2a.set_ylabel('Loss Final')
ax2a.tick_params(axis='x', rotation=45, ha='right')
ax2a.grid(True, alpha=0.3)

for bar, val in zip(bars_final, loss_finals):
    height = bar.get_height()
    ax2a.text(bar.get_x() + bar.get_width()/2., height + 0.001,
              f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

# 2b. Melhoria comparativa
improvements = [result['improvement'] for result in results.values()]
bars_improve = ax2b.bar(optimizer_names, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax2b.set_title('Melhoria no Loss', fontweight='bold', fontsize=14)
ax2b.set_ylabel('Melhoria (Loss inicial - Loss final)')
ax2b.tick_params(axis='x', rotation=45, ha='right')
ax2b.grid(True, alpha=0.3)

for bar, val in zip(bars_improve, improvements):
    height = bar.get_height()
    ax2b.text(bar.get_x() + bar.get_width()/2., height + 0.001,
              f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

# 2c. Acurácia comparativa
accuracies = [result['accuracy'] for result in results.values()]
bars_acc = ax2c.bar(optimizer_names, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax2c.set_title('Acurácia Final', fontweight='bold', fontsize=14)
ax2c.set_ylabel('Acurácia')
ax2c.tick_params(axis='x', rotation=45, ha='right')
ax2c.grid(True, alpha=0.3)

for bar, val in zip(bars_acc, accuracies):
    height = bar.get_height()
    ax2c.text(bar.get_x() + bar.get_width()/2., height + 0.01,
              f'{val:.2%}', ha='center', va='bottom', fontweight='bold')

# 2d. Velocidade de convergência (épocas para atingir 90% da melhoria)
convergence_epochs = []
for name, result in results.items():
    losses = result['losses']
    target_loss = result['initial_loss'] - 0.9 * result['improvement']
    
    for i, loss in enumerate(losses):
        if loss <= target_loss:
            convergence_epochs.append(i)
            break
    else:
        convergence_epochs.append(len(losses))

bars_conv = ax2d.bar(optimizer_names, convergence_epochs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax2d.set_title('Épocas para 90% da Melhoria', fontweight='bold', fontsize=14)
ax2d.set_ylabel('Épocas')
ax2d.tick_params(axis='x', rotation=45, ha='right')
ax2d.grid(True, alpha=0.3)

for bar, val in zip(bars_conv, convergence_epochs):
    height = bar.get_height()
    ax2d.text(bar.get_x() + bar.get_width()/2., height + 1,
              f'{val}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Figura 3: Análise das Predições
fig3, axes = plt.subplots(2, 3, figsize=(18, 12))
fig3.suptitle('Análise das Predições por Otimizador', fontsize=18, fontweight='bold')

for i, (name, result) in enumerate(results.items()):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    # Calcular predições corretas vs incorretas
    predictions = np.argmax(result['predictions'], axis=1)
    correct = predictions == y
    correct_count = np.sum(correct)
    incorrect_count = len(correct) - correct_count
    
    # Gráfico de pizza
    sizes = [correct_count, incorrect_count]
    labels = ['Corretas', 'Incorretas']
    colors_pie = ['#4CAF50', '#F44336']
    
    ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax.set_title(f"{name}\nAcurácia: {correct_count/len(correct):.1%}", 
                fontweight='bold', fontsize=12)

# Remover subplot extra se necessário
if len(results) < 6:
    axes[1, 2].remove()

plt.tight_layout()
plt.show()

# ========== ANÁLISE DETALHADA DOS OTIMIZADORES ==========

print("\n" + "="*80)
print("ANÁLISE DETALHADA DOS OTIMIZADORES")
print("="*80)

print("\n1. SGD (Stochastic Gradient Descent):")
print("   - Mais simples e direto")
print("   - Pode ser lento em vales estreitos")
print("   - Sensível ao learning rate")

print("\n2. SGD + Momentum:")
print("   - Adiciona 'inércia' ao movimento")
print("   - Ajuda a escapar de mínimos locais")
print("   - Acelera convergência em vales")

print("\n3. RMSprop:")
print("   - Adapta learning rate por parâmetro")
print("   - Bom para problemas com gradientes de magnitudes diferentes")
print("   - Estável em diferentes escalas")

print("\n4. Adam:")
print("   - Combina momentum + adaptação de learning rate")
print("   - Geralmente o mais robusto")
print("   - Menos tuning de hyperparâmetros")

print("\n5. AdamW:")
print("   - Adam com weight decay corrigido")
print("   - Melhor para regularização")
print("   - Mais estável em treinamentos longos")

# Resumo final
print("\n" + "="*80)
print("RESUMO COMPARATIVO FINAL")
print("="*80)
print(f"{'Otimizador':<15} {'Loss Final':<12} {'Melhoria':<12} {'Acurácia':<12} {'Convergência':<12}")
print("-" * 80)
for name, result in results.items():
    convergence = convergence_epochs[list(results.keys()).index(name)]
    print(f"{name:<15} {result['final_loss']:<12.4f} {result['improvement']:<12.4f} {result['accuracy']:<12.1%} {convergence:<12}")

print("\n" + "="*80)
print("CONCLUSÕES:")
print("="*80)
print("1. Adam geralmente oferece melhor performance geral")
print("2. SGD + Momentum é uma boa alternativa simples")
print("3. RMSprop é eficiente para gradientes de diferentes escalas")
print("4. AdamW é ideal quando regularização é importante")
print("5. A escolha depende do problema específico e recursos disponíveis!") 