import numpy as np
import matplotlib.pyplot as plt

print("=== VISUALIZAÇÃO DO BACKPROPAGATION ===")
print("Vamos acompanhar como os gradientes fluem pela rede neural\n")

# ========== DEFINIÇÕES NECESSÁRIAS ==========
# Dados de exemplo (mesmos do arquivo 07)
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

# Rede neural simples para demonstração
# 2 inputs -> 3 neurônios (camada 1) -> 2 outputs (camada 2)

# Dados de exemplo para demonstração simples
X_simple = np.array([[1.0, 2.0]])  # 1 exemplo com 2 features
y_true_simple = np.array([[0, 1]])  # Rótulo verdadeiro: [0, 1]

print("DADOS DE ENTRADA:")
print(f"X = {X_simple}")
print(f"y_true = {y_true_simple}")
print()

# Inicializar pesos (valores pequenos para facilitar visualização)
np.random.seed(42)
W1 = np.array([[0.1, 0.2, 0.3],
               [0.4, 0.5, 0.6]])  # 2x3
b1 = np.array([[0.1, 0.1, 0.1]])  # 1x3

W2 = np.array([[0.1, 0.2],
               [0.3, 0.4],
               [0.5, 0.6]])  # 3x2
b2 = np.array([[0.1, 0.1]])  # 1x2

print("PESOS INICIAIS:")
print(f"W1 (camada 1):\n{W1}")
print(f"b1 (camada 1): {b1}")
print(f"W2 (camada 2):\n{W2}")
print(f"b2 (camada 2): {b2}")
print()

# ========== FORWARD PASS ==========
print("=== FORWARD PASS ===")

# Camada 1
z1 = np.dot(X_simple, W1) + b1
print(f"z1 (antes da ativação): {z1}")
a1 = np.maximum(0, z1)  # ReLU
print(f"a1 (após ReLU): {a1}")

# Camada 2
z2 = np.dot(a1, W2) + b2
print(f"z2 (antes da ativação): {z2}")

# Softmax
exp_z2 = np.exp(z2 - np.max(z2))
a2 = exp_z2 / np.sum(exp_z2)
print(f"a2 (após Softmax): {a2}")

# Loss
loss = -np.sum(y_true_simple * np.log(a2 + 1e-15))
print(f"Loss: {loss:.4f}")
print()

# ========== BACKWARD PASS ==========
print("=== BACKWARD PASS (BACKPROPAGATION) ===")

# Learning rate
lr = 0.1

# 1. Gradiente da loss em relação à saída (a2)
d_loss_d_a2 = a2 - y_true_simple
print(f"1. Gradiente da loss em relação à saída:")
print(f"   d_loss_d_a2 = a2 - y_true = {a2} - {y_true_simple} = {d_loss_d_a2}")

# 2. Gradiente da loss em relação a z2 (antes do softmax)
# Para softmax + cross-entropy, a derivada é simples: a2 - y_true
d_loss_d_z2 = d_loss_d_a2
print(f"2. Gradiente da loss em relação a z2:")
print(f"   d_loss_d_z2 = {d_loss_d_z2}")

# 3. Gradientes da camada 2 (W2, b2)
d_loss_d_W2 = np.dot(a1.T, d_loss_d_z2)
d_loss_d_b2 = np.sum(d_loss_d_z2, axis=0, keepdims=True)
print(f"3. Gradientes da camada 2:")
print(f"   d_loss_d_W2 = a1.T × d_loss_d_z2 = {a1.T} × {d_loss_d_z2} = \n{d_loss_d_W2}")
print(f"   d_loss_d_b2 = sum(d_loss_d_z2) = {d_loss_d_b2}")

# 4. Gradiente propagado para a camada 1
d_loss_d_a1 = np.dot(d_loss_d_z2, W2.T)
print(f"4. Gradiente propagado para a1:")
print(f"   d_loss_d_a1 = d_loss_d_z2 × W2.T = {d_loss_d_z2} × {W2.T} = {d_loss_d_a1}")

# 5. Gradiente da loss em relação a z1 (antes da ReLU)
d_loss_d_z1 = d_loss_d_a1 * (z1 > 0)  # Derivada da ReLU
print(f"5. Gradiente da loss em relação a z1:")
print(f"   d_loss_d_z1 = d_loss_d_a1 × (z1 > 0) = {d_loss_d_a1} × {(z1 > 0)} = {d_loss_d_z1}")

# 6. Gradientes da camada 1 (W1, b1)
d_loss_d_W1 = np.dot(X_simple.T, d_loss_d_z1)
d_loss_d_b1 = np.sum(d_loss_d_z1, axis=0, keepdims=True)
print(f"6. Gradientes da camada 1:")
print(f"   d_loss_d_W1 = X.T × d_loss_d_z1 = {X_simple.T} × {d_loss_d_z1} = \n{d_loss_d_W1}")
print(f"   d_loss_d_b1 = sum(d_loss_d_z1) = {d_loss_d_b1}")
print()

# ========== ATUALIZAÇÃO DOS PESOS ==========
print("=== ATUALIZAÇÃO DOS PESOS (GRADIENT DESCENT) ===")

# Atualizar camada 2
W2_new = W2 - lr * d_loss_d_W2
b2_new = b2 - lr * d_loss_d_b2

# Atualizar camada 1
W1_new = W1 - lr * d_loss_d_W1
b1_new = b1 - lr * d_loss_d_b1

print("PESOS ANTES DA ATUALIZAÇÃO:")
print(f"W1:\n{W1}")
print(f"W2:\n{W2}")

print("\nGRADIENTES CALCULADOS:")
print(f"d_loss_d_W1:\n{d_loss_d_W1}")
print(f"d_loss_d_W2:\n{d_loss_d_W2}")

print("\nPESOS APÓS ATUALIZAÇÃO:")
print(f"W1_new = W1 - {lr} × d_loss_d_W1:\n{W1_new}")
print(f"W2_new = W2 - {lr} × d_loss_d_W2:\n{W2_new}")

# ========== VERIFICAÇÃO ==========
print("\n=== VERIFICAÇÃO: FORWARD PASS COM NOVOS PESOS ===")

# Forward pass com novos pesos
z1_new = np.dot(X_simple, W1_new) + b1_new
a1_new = np.maximum(0, z1_new)
z2_new = np.dot(a1_new, W2_new) + b2_new
exp_z2_new = np.exp(z2_new - np.max(z2_new))
a2_new = exp_z2_new / np.sum(exp_z2_new)
loss_new = -np.sum(y_true_simple * np.log(a2_new + 1e-15))

print(f"Loss antes: {loss:.4f}")
print(f"Loss depois: {loss_new:.4f}")
print(f"Melhoria: {loss - loss_new:.4f}")

# ========== VISUALIZAÇÃO GRÁFICA ==========
print("\n=== VISUALIZAÇÃO GRÁFICA DO FLUXO DE GRADIENTES ===")

# Configurar estilo dos gráficos
plt.style.use('default')
plt.rcParams['font.size'] = 12

# ========== FIGURA 1: Estrutura da Rede Neural ==========
print("\n1. Criando diagrama da estrutura da rede neural...")
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
ax1.set_title('Estrutura da Rede Neural e Fluxo de Informação', fontsize=16, fontweight='bold', pad=20)

# Desenhar a estrutura da rede
# Input layer
input_circle1 = plt.Circle((0.1, 0.5), 0.05, color='lightblue', ec='black', linewidth=2)
input_circle2 = plt.Circle((0.1, 0.3), 0.05, color='lightblue', ec='black', linewidth=2)
ax1.add_patch(input_circle1)
ax1.add_patch(input_circle2)
ax1.text(0.1, 0.7, 'Input Layer\n(2 neurônios)', ha='center', va='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

# Hidden layer
hidden_circles = []
for i in range(3):
    circle = plt.Circle((0.4, 0.6 - i*0.2), 0.05, color='lightgreen', ec='black', linewidth=2)
    hidden_circles.append(circle)
    ax1.add_patch(circle)
ax1.text(0.4, 0.9, 'Hidden Layer\n(3 neurônios, ReLU)', ha='center', va='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))

# Output layer
output_circle1 = plt.Circle((0.7, 0.6), 0.05, color='lightcoral', ec='black', linewidth=2)
output_circle2 = plt.Circle((0.7, 0.4), 0.05, color='lightcoral', ec='black', linewidth=2)
ax1.add_patch(output_circle1)
ax1.add_patch(output_circle2)
ax1.text(0.7, 0.9, 'Output Layer\n(2 neurônios, Softmax)', ha='center', va='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))

# Conexões (forward pass)
for i in range(2):
    for j in range(3):
        ax1.arrow(0.15, 0.5 - i*0.2, 0.2, 0.1 - j*0.2, head_width=0.02, head_length=0.02, 
                 fc='blue', ec='blue', alpha=0.6, linewidth=1.5)
for i in range(3):
    for j in range(2):
        ax1.arrow(0.45, 0.6 - i*0.2, 0.2, 0.1 - j*0.2, head_width=0.02, head_length=0.02, 
                 fc='blue', ec='blue', alpha=0.6, linewidth=1.5)

# Legendas
ax1.text(0.85, 0.8, 'Forward Pass\n(azul)', ha='center', va='center', color='blue', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
ax1.text(0.85, 0.6, 'Backward Pass\n(vermelho)', ha='center', va='center', color='red', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
plt.tight_layout()
plt.show()

# ========== FIGURA 2: Valores das Ativações ==========
print("2. Criando gráfico dos valores das ativações...")
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
ax2.set_title('Valores das Ativações por Camada (Forward Pass)', fontsize=16, fontweight='bold', pad=20)

layers = ['Input', 'Hidden (ReLU)', 'Output (Softmax)']
values = [X_simple[0], a1[0], a2[0]]
colors = ['#4CAF50', '#2196F3', '#FF9800']

x_pos = np.arange(len(layers))
bars = ax2.bar(x_pos, [np.mean(v) for v in values], color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# Adicionar valores nas barras
for i, (bar, val) in enumerate(zip(bars, values)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'Média: {height:.3f}\nValores: {[f"{v:.3f}" for v in val]}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Valor de Ativação', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(layers, fontsize=12)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========== FIGURA 3: Gradientes Calculados ==========
print("3. Criando gráfico dos gradientes calculados...")
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
ax3.set_title('Gradientes Calculados (Backward Pass)', fontsize=16, fontweight='bold', pad=20)

grad_names = ['d_W1', 'd_b1', 'd_W2', 'd_b2']
grad_values = [d_loss_d_W1.flatten(), d_loss_d_b1.flatten(), 
               d_loss_d_W2.flatten(), d_loss_d_b2.flatten()]

colors_grad = ['#E91E63', '#9C27B0', '#3F51B5', '#009688']
x_pos_grad = np.arange(len(grad_names))
grad_bars = ax3.bar(x_pos_grad, [np.mean(np.abs(g)) for g in grad_values], 
                    color=colors_grad, alpha=0.8, edgecolor='black', linewidth=2)

# Adicionar valores nas barras
for i, (bar, grads) in enumerate(zip(grad_bars, grad_values)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'Média: {height:.4f}\nValores: {[f"{g:.4f}" for g in grads]}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_ylabel('Magnitude do Gradiente', fontsize=12)
ax3.set_xticks(x_pos_grad)
ax3.set_xticklabels(grad_names, fontsize=12)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========== FIGURA 4: Evolução do Loss ==========
print("4. Criando gráfico da evolução do loss...")
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
ax4.set_title('Evolução do Loss (Antes vs Depois)', fontsize=16, fontweight='bold', pad=20)

comparison_data = ['Antes', 'Depois']
loss_values = [loss, loss_new]
colors_loss = ['#F44336', '#4CAF50']

bars_loss = ax4.bar(comparison_data, loss_values, color=colors_loss, alpha=0.8, edgecolor='black', linewidth=3)

# Adicionar valores nas barras
for bar, val in zip(bars_loss, loss_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{val:.4f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax4.set_ylabel('Loss (Cross-Entropy)', fontsize=12)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========== FIGURA 5: Comparação Detalhada dos Pesos ==========
print("5. Criando gráfico detalhado dos pesos...")
fig5, ax5 = plt.subplots(1, 1, figsize=(14, 8))
ax5.set_title('Comparação Detalhada: Pesos Antes vs Depois da Atualização', fontsize=16, fontweight='bold', pad=20)

# Preparar dados para visualização
weight_names = ['W1_11', 'W1_12', 'W1_13', 'W1_21', 'W1_22', 'W1_23', 
                'W2_11', 'W2_12', 'W2_21', 'W2_22', 'W2_31', 'W2_32']
weights_before = np.concatenate([W1.flatten(), W2.flatten()])
weights_after = np.concatenate([W1_new.flatten(), W2_new.flatten()])

x_pos_weights = np.arange(len(weight_names))
width = 0.35

bars_before = ax5.bar(x_pos_weights - width/2, weights_before, width, 
                      label='Antes da Atualização', color='#FF5722', alpha=0.7, edgecolor='black', linewidth=1)
bars_after = ax5.bar(x_pos_weights + width/2, weights_after, width, 
                     label='Depois da Atualização', color='#4CAF50', alpha=0.7, edgecolor='black', linewidth=1)

# Adicionar valores nas barras
for i, (bar_before, bar_after) in enumerate(zip(bars_before, bars_after)):
    # Valor antes
    ax5.text(bar_before.get_x() + bar_before.get_width()/2., bar_before.get_height() + 0.001,
             f'{weights_before[i]:.3f}', ha='center', va='bottom', fontsize=9, rotation=45)
    # Valor depois
    ax5.text(bar_after.get_x() + bar_after.get_width()/2., bar_after.get_height() + 0.001,
             f'{weights_after[i]:.3f}', ha='center', va='bottom', fontsize=9, rotation=45)

ax5.set_xlabel('Pesos da Rede Neural', fontsize=12)
ax5.set_ylabel('Valor do Peso', fontsize=12)
ax5.set_xticks(x_pos_weights)
ax5.set_xticklabels(weight_names, rotation=45, ha='right', fontsize=10)
ax5.legend(fontsize=12)
ax5.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========== FIGURA 6: Resumo Visual ==========
print("6. Criando resumo visual...")
fig6, ((ax6a, ax6b), (ax6c, ax6d)) = plt.subplots(2, 2, figsize=(12, 10))
fig6.suptitle('Resumo Visual do Backpropagation', fontsize=18, fontweight='bold')

# 6a. Loss antes vs depois (versão compacta)
ax6a.bar(['Antes', 'Depois'], [loss, loss_new], color=['#F44336', '#4CAF50'], alpha=0.8)
ax6a.set_title('Evolução do Loss', fontweight='bold')
ax6a.set_ylabel('Loss')
ax6a.grid(True, alpha=0.3)
for i, v in enumerate([loss, loss_new]):
    ax6a.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# 6b. Gradientes médios
grad_means = [np.mean(np.abs(d_loss_d_W1)), np.mean(np.abs(d_loss_d_b1)), 
              np.mean(np.abs(d_loss_d_W2)), np.mean(np.abs(d_loss_d_b2))]
ax6b.bar(['d_W1', 'd_b1', 'd_W2', 'd_b2'], grad_means, color=['#E91E63', '#9C27B0', '#3F51B5', '#009688'], alpha=0.8)
ax6b.set_title('Gradientes Médios', fontweight='bold')
ax6b.set_ylabel('Magnitude')
ax6b.grid(True, alpha=0.3)

# 6c. Mudança nos pesos
weight_changes = np.abs(weights_after - weights_before)
ax6c.bar(range(len(weight_changes)), weight_changes, color='orange', alpha=0.8)
ax6c.set_title('Mudança Absoluta nos Pesos', fontweight='bold')
ax6c.set_ylabel('|ΔPeso|')
ax6c.set_xlabel('Índice do Peso')
ax6c.grid(True, alpha=0.3)

# 6d. Ativações médias
activation_means = [np.mean(X_simple[0]), np.mean(a1[0]), np.mean(a2[0])]
ax6d.bar(['Input', 'Hidden', 'Output'], activation_means, color=['#4CAF50', '#2196F3', '#FF9800'], alpha=0.8)
ax6d.set_title('Ativações Médias', fontweight='bold')
ax6d.set_ylabel('Valor Médio')
ax6d.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Resumo numérico adicional
print("\n" + "="*60)
print("RESUMO NUMÉRICO DO BACKPROPAGATION")
print("="*60)
print(f"Loss inicial: {loss:.6f}")
print(f"Loss final:   {loss_new:.6f}")
print(f"Melhoria:     {loss - loss_new:.6f}")
print(f"Redução %:    {((loss - loss_new) / loss * 100):.2f}%")
print("\nGradientes médios:")
print(f"  d_W1: {np.mean(np.abs(d_loss_d_W1)):.6f}")
print(f"  d_b1: {np.mean(np.abs(d_loss_d_b1)):.6f}")
print(f"  d_W2: {np.mean(np.abs(d_loss_d_W2)):.6f}")
print(f"  d_b2: {np.mean(np.abs(d_loss_d_b2)):.6f}")
print("\nMudança média nos pesos:")
print(f"  W1: {np.mean(np.abs(W1_new - W1)):.6f}")
print(f"  W2: {np.mean(np.abs(W2_new - W2)):.6f}")

print("\n=== RESUMO DO BACKPROPAGATION ===")
print("1. Forward Pass: Calcula saída e loss")
print("2. Backward Pass: Calcula gradientes usando regra da cadeia")
print("3. Atualização: Ajusta pesos usando gradient descent")
print("4. Resultado: Loss diminui, rede aprende!")

# ========== COMPARAÇÕES DE DIFERENTES ABORDAGENS ==========
print("\n" + "="*80)
print("COMPARAÇÃO DE DIFERENTES ABORDAGENS DE TREINAMENTO")
print("="*80)

# Função para treinar rede com configurações específicas
def train_network(X, y_true, config):
    """Treina uma rede neural com configurações específicas"""
    np.random.seed(42)  # Mesma seed para comparação justa
    
    # Configurar arquitetura
    if config['type'] == 'original':
        # Rede original: 2 camadas
        layer1 = Layer_Dense(X.shape[1], 20)
        layer2 = Layer_Dense(20, 4)
        layers = [layer1, layer2]
    elif config['type'] == 'larger':
        # Rede maior: 4 camadas
        layer1 = Layer_Dense(X.shape[1], 32)
        layer2 = Layer_Dense(32, 24)
        layer3 = Layer_Dense(24, 16)
        layer4 = Layer_Dense(16, 4)
        layers = [layer1, layer2, layer3, layer4]
    
    losses = []
    learning_rate = config['lr']
    
    for epoch in range(config['epochs']):
        # Learning rate adaptativo
        if config['adaptive_lr']:
            learning_rate = config['lr'] * (1 / (1 + 0.01 * epoch))
        
        # Forward pass
        current_input = X
        activations = [current_input]
        
        for i, layer in enumerate(layers[:-1]):
            layer.forward(current_input)
            if config['type'] == 'larger':
                current_input = relu(layer.output)
            else:
                current_input = relu(layer.output)
            activations.append(current_input)
        
        # Última camada (softmax)
        layers[-1].forward(current_input)
        output = softmax(layers[-1].output)
        
        # Loss
        loss = cross_entropy_loss(output, y_true)
        losses.append(loss)
        
        # Backward pass
        dloss = cross_entropy_derivative(output, y_true)
        
        # Backpropagation através das camadas
        for i in range(len(layers) - 1, -1, -1):
            if i == len(layers) - 1:
                layers[i].backward(dloss)
            else:
                if config['type'] == 'larger':
                    drelu = layers[i+1].dinputs * relu_derivative(layers[i].output)
                else:
                    drelu = layers[i+1].dinputs * relu_derivative(layers[i].output)
                layers[i].backward(drelu)
        
        # Atualizar pesos
        for layer in layers:
            layer.weights -= learning_rate * layer.dweights
            layer.biases -= learning_rate * layer.dbiases
    
    # Forward pass final para obter predições
    current_input = X
    for i, layer in enumerate(layers[:-1]):
        layer.forward(current_input)
        current_input = relu(layer.output)
    
    layers[-1].forward(current_input)
    final_output = softmax(layers[-1].output)
    
    return losses, final_output, layers

# Configurações para comparação
configs = {
    'original': {
        'type': 'original',
        'epochs': 1000,
        'lr': 0.1,
        'adaptive_lr': False,
        'name': 'Original (1000 épocas, lr=0.1)'
    },
    'more_epochs': {
        'type': 'original',
        'epochs': 5000,
        'lr': 0.1,
        'adaptive_lr': False,
        'name': 'Mais Épocas (5000 épocas, lr=0.1)'
    },
    'adaptive_lr': {
        'type': 'original',
        'epochs': 1000,
        'lr': 0.1,
        'adaptive_lr': True,
        'name': 'Learning Rate Adaptativo (1000 épocas)'
    },
    'larger_network': {
        'type': 'larger',
        'epochs': 1000,
        'lr': 0.1,
        'adaptive_lr': False,
        'name': 'Rede Maior (4 camadas, 1000 épocas)'
    }
}

# Treinar todas as configurações
results = {}
for key, config in configs.items():
    print(f"\nTreinando: {config['name']}")
    losses, predictions, layers = train_network(X_norm, y_one_hot, config)
    results[key] = {
        'losses': losses,
        'predictions': predictions,
        'config': config,
        'final_loss': losses[-1],
        'initial_loss': losses[0],
        'improvement': losses[0] - losses[-1]
    }
    print(f"  Loss inicial: {losses[0]:.4f}")
    print(f"  Loss final: {losses[-1]:.4f}")
    print(f"  Melhoria: {losses[0] - losses[-1]:.4f}")

# ========== VISUALIZAÇÕES COMPARATIVAS ==========
print("\nCriando visualizações comparativas...")

# Figura 1: Comparação da evolução do loss
fig_comp1, ax_comp1 = plt.subplots(1, 1, figsize=(12, 8))
ax_comp1.set_title('Comparação da Evolução do Loss', fontsize=16, fontweight='bold', pad=20)

colors_comp = ['#FF5722', '#2196F3', '#4CAF50', '#9C27B0']
for i, (key, result) in enumerate(results.items()):
    ax_comp1.plot(result['losses'], label=result['config']['name'], 
                 color=colors_comp[i], linewidth=2, alpha=0.8)

ax_comp1.set_xlabel('Época', fontsize=12)
ax_comp1.set_ylabel('Loss (Cross-Entropy)', fontsize=12)
ax_comp1.legend(fontsize=11)
ax_comp1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Figura 2: Comparação final dos resultados
fig_comp2, ((ax_comp2a, ax_comp2b), (ax_comp2c, ax_comp2d)) = plt.subplots(2, 2, figsize=(15, 12))
fig_comp2.suptitle('Comparação Detalhada das Abordagens', fontsize=18, fontweight='bold')

# 2a. Loss final comparativo
loss_finals = [result['final_loss'] for result in results.values()]
config_names = [result['config']['name'] for result in results.values()]

bars_final = ax_comp2a.bar(config_names, loss_finals, color=colors_comp, alpha=0.8, edgecolor='black', linewidth=2)
ax_comp2a.set_title('Loss Final Comparativo', fontweight='bold', fontsize=14)
ax_comp2a.set_ylabel('Loss Final')
ax_comp2a.tick_params(axis='x', rotation=45, ha='right')
ax_comp2a.grid(True, alpha=0.3)

# Adicionar valores nas barras
for bar, val in zip(bars_final, loss_finals):
    height = bar.get_height()
    ax_comp2a.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

# 2b. Melhoria comparativa
improvements = [result['improvement'] for result in results.values()]
bars_improve = ax_comp2b.bar(config_names, improvements, color=colors_comp, alpha=0.8, edgecolor='black', linewidth=2)
ax_comp2b.set_title('Melhoria no Loss', fontweight='bold', fontsize=14)
ax_comp2b.set_ylabel('Melhoria (Loss inicial - Loss final)')
ax_comp2b.tick_params(axis='x', rotation=45, ha='right')
ax_comp2b.grid(True, alpha=0.3)

for bar, val in zip(bars_improve, improvements):
    height = bar.get_height()
    ax_comp2b.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

# 2c. Acurácia comparativa
accuracies = []
for key, result in results.items():
    predictions = np.argmax(result['predictions'], axis=1)
    accuracy = np.mean(predictions == y)
    accuracies.append(accuracy)

bars_acc = ax_comp2c.bar(config_names, accuracies, color=colors_comp, alpha=0.8, edgecolor='black', linewidth=2)
ax_comp2c.set_title('Acurácia Final', fontweight='bold', fontsize=14)
ax_comp2c.set_ylabel('Acurácia')
ax_comp2c.tick_params(axis='x', rotation=45, ha='right')
ax_comp2c.grid(True, alpha=0.3)

for bar, val in zip(bars_acc, accuracies):
    height = bar.get_height()
    ax_comp2c.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.2%}', ha='center', va='bottom', fontweight='bold')

# 2d. Velocidade de convergência (épocas para atingir 90% da melhoria)
convergence_epochs = []
for key, result in results.items():
    losses = result['losses']
    target_loss = result['initial_loss'] - 0.9 * result['improvement']
    
    # Encontrar primeira época que atinge o target
    for i, loss in enumerate(losses):
        if loss <= target_loss:
            convergence_epochs.append(i)
            break
    else:
        convergence_epochs.append(len(losses))

bars_conv = ax_comp2d.bar(config_names, convergence_epochs, color=colors_comp, alpha=0.8, edgecolor='black', linewidth=2)
ax_comp2d.set_title('Épocas para 90% da Melhoria', fontweight='bold', fontsize=14)
ax_comp2d.set_ylabel('Épocas')
ax_comp2d.tick_params(axis='x', rotation=45, ha='right')
ax_comp2d.grid(True, alpha=0.3)

for bar, val in zip(bars_conv, convergence_epochs):
    height = bar.get_height()
    ax_comp2d.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Figura 3: Análise detalhada das predições
fig_comp3, axes = plt.subplots(2, 2, figsize=(15, 12))
fig_comp3.suptitle('Análise das Predições por Abordagem', fontsize=18, fontweight='bold')

for i, (key, result) in enumerate(results.items()):
    row = i // 2
    col = i % 2
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
    ax.set_title(f"{result['config']['name']}\nAcurácia: {correct_count/len(correct):.1%}", 
                fontweight='bold', fontsize=12)

plt.tight_layout()
plt.show()

# Resumo final
print("\n" + "="*80)
print("RESUMO COMPARATIVO FINAL")
print("="*80)
print(f"{'Abordagem':<35} {'Loss Final':<12} {'Melhoria':<12} {'Acurácia':<12} {'Convergência':<12}")
print("-" * 80)
for key, result in results.items():
    predictions = np.argmax(result['predictions'], axis=1)
    accuracy = np.mean(predictions == y)
    convergence = convergence_epochs[list(results.keys()).index(key)]
    print(f"{result['config']['name']:<35} {result['final_loss']:<12.4f} {result['improvement']:<12.4f} {accuracy:<12.1%} {convergence:<12}")

print("\n" + "="*80)
print("CONCLUSÕES:")
print("="*80)
print("1. Mais épocas geralmente melhoram o resultado, mas com retornos decrescentes")
print("2. Learning rate adaptativo pode acelerar a convergência")
print("3. Redes maiores têm mais capacidade, mas podem overfitar")
print("4. A melhor abordagem depende do problema específico!") 