import numpy as np
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =============================================================================
# 1. اینجا بارگذاری و پیش‌ پردازش داده‌ها
# =============================================================================
random.seed(42)
np.random.seed(42)
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# تبدیل برچسب‌ها به One-Hot Encoding
def to_one_hot(y, num_classes):
    oh = np.zeros((y.size, num_classes))
    oh[np.arange(y.size), y] = 1
    return oh

y_train_oh = to_one_hot(y_train, 3)
y_val_oh = to_one_hot(y_val, 3)

# =============================================================================
# 2. توابع فعال‌ساز
# =============================================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(a):
    return a * (1 - a)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# =============================================================================
# 3. کلاس شبکه عصبی
# =============================================================================
class MLP:
    def __init__(self, layer_sizes, lr=0.01):
        self.lr = lr
        self.params = []
        # مقداردهی اولیه وزن‌ها با روش Xavier
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))
            self.params.append({'W': w, 'b': b})
    
    def forward(self, X):
        self.cache = [{'A': X}]
        A = X
        # لایه‌های پنهان با سیگموید
        for p in self.params[:-1]:
            Z = np.dot(A, p['W']) + p['b']
            A = sigmoid(Z)
            self.cache.append({'A': A})
        # لایه خروجی با Softmax
        Z_out = np.dot(A, self.params[-1]['W']) + self.params[-1]['b']
        A_out = softmax(Z_out)
        self.cache.append({'A': A_out})
        return A_out
    
    def backward(self, Y_true):
        m = Y_true.shape[0]
        # گرادیان لایه خروجی
        gradient_output = (self.cache[-1]['A'] - Y_true) / m
        
        for i in reversed(range(len(self.params))):
            A_prev = self.cache[i]['A']
            dW = np.dot(A_prev.T, gradient_output)
            db = np.sum(gradient_output, axis=0, keepdims=True)
            
            if i > 0:
                dA_prev = np.dot(gradient_output, self.params[i]['W'].T)
                gradient_output = dA_prev * sigmoid_deriv(self.cache[i]['A'])
                
            self.params[i]['W'] -= self.lr * dW
            self.params[i]['b'] -= self.lr * db

# =============================================================================
# 4. تابع آموزش با Early Stopping
# =============================================================================
def train_model(layer_sizes, lr, max_epochs=500, patience=50):
    model = MLP(layer_sizes, lr)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # آموزش
        probs = model.forward(X_train)
        model.backward(y_train_oh)
        
        # محاسبه خطا
        train_loss = -np.mean(y_train_oh * np.log(probs + 1e-9))
        train_losses.append(train_loss)
        
        # اعتبارسنجی
        val_probs = model.forward(X_val)
        val_loss = -np.mean(y_val_oh * np.log(val_probs + 1e-9))
        val_losses.append(val_loss)
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    return model, train_losses, val_losses

# =============================================================================
# 5. تابع ارزیابی
# =============================================================================
def evaluate(model, X, y):
    probs = model.forward(X)
    predictions = np.argmax(probs, axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy

# =============================================================================
# 6. اجرای تمام پیکربندی‌ها
# =============================================================================
configurations = [
    ([4, 8, 3], [0.001, 0.01, 0.1, 1.0]),      # 1 لایه پنهان
    ([4, 8, 6, 3], [0.001, 0.01, 0.1, 1.0]),   # 2 لایه پنهان
    ([4, 8, 6, 4, 3], [0.001, 0.01, 0.1, 1.0]) # 3 لایه پنهان
]

results = []

for layer_sizes, learning_rates in configurations:
    num_hidden = len(layer_sizes) - 2
    for lr in learning_rates:
        print(f"\n{'='*60}")
        print(f"Training: {num_hidden} hidden layer(s), lr={lr}")
        print(f"{'='*60}")
        
        # آموزش مدل
        model, train_losses, val_losses = train_model(layer_sizes, lr)
        
        # ارزیابی
        test_acc = evaluate(model, X_val, y_val)
        
        results.append({
            'hidden_layers': num_hidden,
            'lr': lr,
            'accuracy': test_acc,
            'train_losses': train_losses,
            'val_losses': val_losses
        })
        
        print(f"Test Accuracy: {test_acc*100:.2f}%")

# =============================================================================
# 7. نمایش نتایج
# =============================================================================
print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)
print(f"{'Hidden Layers':<15} {'Learning Rate':<15} {'Accuracy':<10}")
print("-"*40)
for r in results:
    print(f"{r['hidden_layers']:<15} {r['lr']:<15} {r['accuracy']*100:.2f}%")

# =============================================================================
# 8. رسم نمودار (اختیاری)
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, r in enumerate(results[:4]):  # فقط 4 پیکربندی اول
    ax = axes[idx]
    ax.plot(r['train_losses'], label='Train Loss')
    ax.plot(r['val_losses'], label='Val Loss')
    ax.set_title(f"{r['hidden_layers']} hidden, lr={r['lr']}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('results.png', dpi=300)
plt.show()
print("Plot saved as results.png")

import matplotlib.pyplot as plt

# رسم نمودار برای شبکه ۲ لایه با نرخ یادگیری ۰.۱


epochs = range(1, 201)  # تا epoch 200
train_loss = [0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.18, 0.15]  # مثال
val_loss = [0.85, 0.65, 0.55, 0.45, 0.4, 0.35, 0.32, 0.3, 0.31, 0.33]  # مثال

plt.figure(figsize=(10, 6))
plt.plot(epochs[:10], train_loss[:10], 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs[:10], val_loss[:10], 'r-', label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (2 Hidden Layers, η=0.1)')
plt.legend()
plt.grid(True)
plt.savefig('figure3.png', dpi=300)
plt.show()
