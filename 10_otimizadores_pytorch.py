import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

print("=== ESTUDO DOS OTIMIZADORES COM PYTORCH ===")
print("Implementando e comparando diferentes otimizadores usando PyTorch\n")

# ========== CONFIGURAÇÃO INICIAL ==========
torch.manual_seed(42)
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

# Converter para tensores PyTorch
X_tensor = torch.FloatTensor(X_norm)
y_tensor = torch.LongTensor(y)

print("Dados convertidos para tensores PyTorch:")
print(f"X shape: {X_tensor.shape}")
print(f"y shape: {y_tensor.shape}")
print()

# ========== REDE NEURAL COM PYTORCH ==========

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.softmax(self.layer2(x))
        return x

# ========== FUNÇÃO DE TREINAMENTO COM OTIMIZADORES ==========

def train_with_optimizer(X, y, optimizer_class, optimizer_params, epochs=1000):
    """Treina uma rede neural usando um otimizador específico do PyTorch"""
    torch.manual_seed(42)  # Mesma seed para comparação justa
    
    # Criar rede neural
    model = SimpleNeuralNetwork(input_size=5, hidden_size=20, output_size=4)
    
    # Criar otimizador
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    
    # Função de loss
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        losses.append(loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Forward pass final para obter predições
    with torch.no_grad():
        final_outputs = model(X)
        predictions = final_outputs.numpy()
    
    return losses, predictions

# ========== COMPARAÇÃO DOS OTIMIZADORES ==========

print("=== COMPARAÇÃO DOS OTIMIZADORES PYTORCH ===")

# Configurar otimizadores com suas classes e parâmetros
optimizer_configs = {
    'SGD': (optim.SGD, {'lr': 0.1}),
    'SGD + Momentum': (optim.SGD, {'lr': 0.1, 'momentum': 0.9}),
    'RMSprop': (optim.RMSprop, {'lr': 0.001, 'alpha': 0.9}),
    'Adam': (optim.Adam, {'lr': 0.001, 'betas': (0.9, 0.999)}),
    'AdamW': (optim.AdamW, {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0.01})
}

# Treinar com cada otimizador
results = {}
for name, (optimizer_class, params) in optimizer_configs.items():
    print(f"\nTreinando com {name}...")
    losses, predictions = train_with_optimizer(X_tensor, y_tensor, optimizer_class, params, epochs=1000)
    
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
ax1.set_title('Comparação da Evolução do Loss por Otimizador (PyTorch)', fontsize=16, fontweight='bold', pad=20)

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
fig2.suptitle('Análise Detalhada dos Otimizadores (PyTorch)', fontsize=18, fontweight='bold')

# 2a. Loss final comparativo
loss_finals = [result['final_loss'] for result in results.values()]
optimizer_names = list(results.keys())

bars_final = ax2a.bar(optimizer_names, loss_finals, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax2a.set_title('Loss Final Comparativo', fontweight='bold', fontsize=14)
ax2a.set_ylabel('Loss Final')
ax2a.tick_params(axis='x', rotation=45)
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
ax2b.tick_params(axis='x', rotation=45)
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
ax2c.tick_params(axis='x', rotation=45)
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
ax2d.tick_params(axis='x', rotation=45)
ax2d.grid(True, alpha=0.3)

for bar, val in zip(bars_conv, convergence_epochs):
    height = bar.get_height()
    ax2d.text(bar.get_x() + bar.get_width()/2., height + 1,
              f'{val}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Figura 3: Análise das Predições
fig3, axes = plt.subplots(2, 3, figsize=(18, 12))
fig3.suptitle('Análise das Predições por Otimizador (PyTorch)', fontsize=18, fontweight='bold')

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

# ========== COMPARAÇÃO COM IMPLEMENTAÇÃO MANUAL ==========

print("\n" + "="*80)
print("COMPARAÇÃO: PYTORCH vs IMPLEMENTAÇÃO MANUAL")
print("="*80)

print("\nVantagens do PyTorch:")
print("1. Código muito mais limpo e conciso")
print("2. Otimizadores já implementados e otimizados")
print("3. Autograd automático (não precisa implementar backpropagation)")
print("4. GPU support nativo")
print("5. Comunidade ativa e documentação excelente")

print("\nDiferenças na implementação:")
print("- PyTorch: ~50 linhas de código")
print("- Manual: ~200 linhas de código")
print("- PyTorch: Otimizadores prontos")
print("- Manual: Implementação do zero")

# Resumo final
print("\n" + "="*80)
print("RESUMO COMPARATIVO FINAL (PYTORCH)")
print("="*80)
print(f"{'Otimizador':<15} {'Loss Final':<12} {'Melhoria':<12} {'Acurácia':<12} {'Convergência':<12}")
print("-" * 80)
for name, result in results.items():
    convergence = convergence_epochs[list(results.keys()).index(name)]
    print(f"{name:<15} {result['final_loss']:<12.4f} {result['improvement']:<12.4f} {result['accuracy']:<12.1%} {convergence:<12}")

print("\n" + "="*80)
print("CONCLUSÕES:")
print("="*80)
print("1. PyTorch simplifica drasticamente a implementação")
print("2. Mesmos resultados com muito menos código")
print("3. Otimizadores já otimizados e testados")
print("4. Pronto para produção e pesquisa")
print("5. Framework recomendado para projetos reais!") 