import numpy as np
import matplotlib.pyplot as plt

# Гаусс
def solve_gauss(matrix_A, vector_b, eps=1e-12):
    A = matrix_A.astype(float)
    b = vector_b.astype(float).reshape(-1, 1)
    n = len(b)
    system = np.hstack((A, b))
    for i in range(n):
        pivot = np.abs(system[i:, i]).argmax() + i
        if np.abs(system[pivot, i]) < eps:
            raise ValueError("Матрица вырождена")
        system[[i, pivot]] = system[[pivot, i]]
        system[i] = system[i] / system[i, i]
        for j in range(i + 1, n):
            system[j] = system[j] - system[j, i] * system[i]
    for i in range(n - 1, 0, -1):
        for j in range(0, i):
            system[j] = system[j] - system[j, i] * system[i]
    return system[:, -1]

# МНК
def least_squares(x, y, m):
    n = len(x)
    A = np.zeros((n, m + 1))
    for i in range(n):
        for j in range(m + 1):
            A[i, j] = x[i] ** j
    B = A.T @ A
    d = A.T @ y
    return solve_gauss(B, d)


def load_data_1(filename):
    data = np.loadtxt(filename, skiprows=1)
    return data[:, 0], data[:, 1]

def load_data_2(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        x = np.array([float(val) for val in lines[0].split()])
        y = np.array([float(val) for val in lines[1].split()])
    return x, y

# графики
def plot_results(x, y, title):
    c1 = least_squares(x, y, 1)
    c2 = least_squares(x, y, 2)
    
    plt.scatter(x, y, color='lightgray', s=15, label='Данные', zorder=1)
    x_range = np.linspace(min(x), max(x), 100)
    
    if c1 is not None:
        y1 = c1[0] + c1[1] * x_range
        plt.plot(x_range, y1, label=f'm=1: y={c1[0]:.2f}+{c1[1]:.2f}x', color='blue', zorder=2)
        
    if c2 is not None:
        y2 = c2[0] + c2[1] * x_range + c2[2] * (x_range**2)
        
        plt.plot(x_range, y2, '--', label=f'm=2: y={c2[0]:.2f}+{c2[1]:.2f}x+{c2[2]:.4f}x²', 
                 color='red', linewidth=2, zorder=3)
        
    plt.title(title)
    plt.legend(fontsize='x-small')
    plt.grid(True, alpha=0.2)


plt.figure(figsize=(18, 5))

# 1 набор
try:
    x1, y1 = load_data_1('first_set.txt')
    plt.subplot(1, 3, 1)
    plot_results(x1, y1, "Набор №1")
except: print("Файл first_set.txt не найден")

# 2 набор
try:
    x2, y2 = load_data_2('secnd_set.txt')
    plt.subplot(1, 3, 2)
    plot_results(x2, y2, "Набор №2")
except: print("Файл secnd_set.txt не найден")

# 3 набор (мой)
try:
    x3, y3 = load_data_1('mine_set.txt')
    plt.subplot(1, 3, 3)
    plot_results(x3, y3, "Сгенерированный набор (Кривая)")
except: print("Файл mine_set.txt не найден")

plt.tight_layout()
plt.show()