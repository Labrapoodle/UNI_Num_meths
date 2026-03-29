import numpy as np
import matplotlib.pyplot as plt

def func(x, func_number, deriviate="bare"):
    x = np.array(x)
    if func_number == "first":
        if deriviate == "bare":   return np.sin(x) + x**2
        if deriviate == "sec":    return np.sin(x) * (2 - x**2) + 4 * x * np.cos(x)
        if deriviate == "fourth": return (x**2 - 12) * np.sin(x) - 8 * np.cos(x)
    elif func_number == "second":
        if deriviate == "bare":   return np.exp(-x) * np.cos(x)
        if deriviate == "sec":    return 2 * np.exp(-x) * np.sin(x)
        if deriviate == "fourth": return -4 * np.exp(-x) * np.cos(x)
    elif func_number == "third":
        if deriviate == "bare":   return 1 / (1 + x**2)
        if deriviate == "sec":    return 8*x**2 / (1+x**2)**3 - 2 / (1+x**2)**2
        if deriviate == "fourth": return -288*x**2 / (1+x**2)**4 + 24 / (1+x**2)**3 + 384*x**4 / (1+x**2)**5
    raise ValueError("Ошибка параметров")

def analytical_integral(func_num, a, b):
    if func_num == "first":
        f = lambda x: -np.cos(x) + (x**3 / 3)
    elif func_num == "second":
        f = lambda x: (np.exp(-x) * (np.sin(x) - np.cos(x))) / 2
    elif func_num == "third":
        f = lambda x: np.arctan(x)
    else: return 0
    return f(b) - f(a)

def trapez(a, b, n, func_num):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x, func_num, "bare")
    I = h * (0.5 * (y[0] + y[-1]) + np.sum(y[1:-1]))
    max_sec = np.max(np.abs(func(x, func_num, "sec")))
    error = (h**2) * (b - a) * max_sec / 12
    return I, error

def simps(a, b, n, func_num):
    if n % 2 != 0: n += 1 
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x, func_num, "bare")
    I = (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))
    max_fourth = np.max(np.abs(func(x, func_num, "fourth")))
    error = (h**4) * (b - a) * max_fourth / 180
    return I, error

# Список n
n_list = list(range(1, 11)) + list(range(15, 51, 5)) + list(range(60, 101, 10))

tasks = [
    ("first", -5, 5, "sin(x) + x^2"),
    ("second", -4, -2, "exp(-x) * cos(x)"),
    ("third", -4, 4, "1 / (1 + x^2)")
]

fig, axes = plt.subplots(3, 1, figsize=(10, 14))
fig.tight_layout(pad=6.0)

for i, (f_num, a, b, f_name) in enumerate(tasks):
    exact = analytical_integral(f_num, a, b)
    print(f"\n=== ФУНКЦИЯ: {f_name} на [{a}, {b}] ===")
    print(f"АНАЛИТИЧЕСКОЕ ЗНАЧЕНИЕ: {exact:.10f}")
    print(f"{'n':>4} | {'Trapez I':>12} | {'Tr. Err':>10} | {'Simps I':>12} | {'Sim. Err':>10}")
    print("-" * 65)
    
    for n_val in n_list:
        i_tr, e_tr = trapez(a, b, n_val, f_num)
        i_si, e_si = simps(a, b, n_val, f_num)
        print(f"{n_val:4d} | {i_tr:12.6f} | {e_tr:10.1e} | {i_si:12.6f} | {e_si:10.1e}")

    x_plot = np.linspace(a, b, 300)
    y_plot = func(x_plot, f_num, "bare")
    axes[i].plot(x_plot, y_plot, label=f"f(x) = {f_name}", color='teal', lw=2)
    axes[i].fill_between(x_plot, y_plot, color='teal', alpha=0.1)
    axes[i].set_title(f"Интеграл: {f_name} (Точно: {exact:.4f})")
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

plt.show()