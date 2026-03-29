import numpy as np
import matplotlib.pyplot as plt

# 1. Лагранж
def lagrange_interpolation(x_nodes, y_nodes, x):
    n = len(x_nodes)
    res = 0
    for i in range(n):
        # базисный полином l_i(x)
        p = 1
        for j in range(n):
            if i != j:
                p *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        res += y_nodes[i] * p
    return res

# 2. Ньютона
def newton_interpolation(x_nodes, y_nodes, x):
    n = len(x_nodes)
    
    coef = np.copy(y_nodes).astype(float)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x_nodes[i] - x_nodes[i-j])
    
    
    res = coef[n-1]
    for i in range(n - 2, -1, -1):
        res = res * (x - x_nodes[i]) + coef[i]
    return res


np.random.seed(42) 
x_nodes = np.sort(np.random.uniform(-10, 10, 7))
y_nodes = np.sin(x_nodes / 2) * 5  # Этого здесь нет!


x_target = 2.5


val_lagrange = lagrange_interpolation(x_nodes, y_nodes, x_target)
val_newton = newton_interpolation(x_nodes, y_nodes, x_target)

print(f"Результаты для x = {x_target}:")
print(f"Полином Лагранжа L({x_target}) = {val_lagrange:.4f}")
print(f"Полином Ньютона  N({x_target}) = {val_newton:.4f}")


a, b, h = -10, 10, 0.1
x_range = np.arange(a, b + h, h)


y_lagrange_plot = [lagrange_interpolation(x_nodes, y_nodes, xi) for xi in x_range]

plt.figure(figsize=(10, 6))
plt.plot(x_range, y_lagrange_plot, label='Интерполяционный полином', color='blue', alpha=0.7)
plt.scatter(x_nodes, y_nodes, color='red', zorder=5, label='Узлы интерполяции (in.txt)')
plt.scatter([x_target], [val_lagrange], color='green', marker='x', s=100, label=f'Точка x={x_target}')

plt.title(f"Интерполяция на отрезке [{a}, {b}] с шагом {h}")
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()