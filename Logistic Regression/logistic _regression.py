import numpy as np

# 1. Create synthetic binary data
np.random.seed(42)
X = np.linspace(-10, 10, 100)
y = (X > 0).astype(int)  # class 1 if x > 0, else 0

# 2. Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 3. Initialize parameters
w = 0.0
b = 0.0
lr = 0.1
epochs = 1000

# 4. Training loop
for epoch in range(epochs):
    z = w * X + b
    y_hat = sigmoid(z)
    
    # Binary cross entropy loss
    loss = -np.mean(y * np.log(y_hat + 1e-9) + (1-y) * np.log(1 - y_hat + 1e-9))
    
    # Gradients
    dw = np.mean((y_hat - y) * X)
    db = np.mean(y_hat - y)
    
    # Update parameters
    w -= lr * dw
    b -= lr * db
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss={loss:.4f}, w={w:.4f}, b={b:.4f}")
        


import matplotlib.pyplot as plt

plt.scatter(X, y, label="Data")
plt.plot(X, sigmoid(w*X+b), color="red", label="Sigmoid Boundary")
plt.legend()
plt.show()