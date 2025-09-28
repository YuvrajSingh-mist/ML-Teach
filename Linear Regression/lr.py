import numpy as np
import matplotlib.pyplot as plt

# 1. Create some fake data (y = 2x + 3 + noise)
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 3 + np.random.randn(100) * 2  # add noise

# 2. Initialize parameters
w = 0.0
b = 0.0
lr = 0.01  # learning rate
epochs = 1000

# 3. Training loop (gradient descent)
for epoch in range(epochs):
    y_hat = w * X + b
    
    # Loss (MSE)
    loss = np.mean((y - y_hat) ** 2)
    
    # Gradients
    dw = -2 * np.mean(X * (y - y_hat))
    db = -2 * np.mean(y - y_hat)
    
    # Update parameters
    w -= lr * dw
    b -= lr * db
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss={loss:.4f}, w={w:.4f}, b={b:.4f}")
        

plt.scatter(X, y, label="Data")
plt.plot(X, w*X+b, color="red", label="Learned Line")
plt.legend()
plt.show()