import numpy as np

# Определение функции и её производной
def f(x):
    return 2 - np.sqrt(x**3) - 2 * np.log(x)

def df(x):
    return -1.5 * np.sqrt(x) - 2 / x

# 1. Метод бисекций (деления пополам)
def bisection_method(a, b, eps):
    if f(a) * f(b) >= 0:
        return None, 0
    
    iter_count = 0
    while (b - a) / 2 > eps:
        iter_count += 1
        c = (a + b) / 2
        if f(c) == 0:
            return c, iter_count
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2, iter_count

# 2. Метод Ньютона (касательных)
def newton_method(x0, eps):
    iter_count = 0
    x_curr = x0
    while True:
        iter_count += 1
        x_next = x_curr - f(x_curr) / df(x_curr)
        if abs(x_next - x_curr) < eps:
            return x_next, iter_count
        x_curr = x_next

# 3. Метод хорд
def secant_method(a, b, eps):
    iter_count = 0
    x_prev = a
    x_curr = b
    while True:
        iter_count += 1
        # Формула: x_next = x_curr - f(x_curr) * (x_curr - x_prev) / (f(x_curr) - f(x_prev))
        x_next = x_curr - f(x_curr) * (x_curr - x_prev) / (f(x_curr) - f(x_prev))
        if abs(x_next - x_curr) < eps:
            return x_next, iter_count
        x_prev, x_curr = x_curr, x_next

# Параметры задачи
a, b = 1.0, 2.0
eps = 1e-10


res_b, it_b = bisection_method(a, b, eps)
res_n, it_n = newton_method(b, eps) # Начинаем с b, т.к. f(b)*f''(b) > 0
res_s, it_s = secant_method(a, b, eps)

print(f"{'Метод':<15} | {'Корень':<12} | {'Итерации'}")
print("-" * 40)
print(f"{'Бисекции':<15} | {res_b:<12.7f} | {it_b}")
print(f"{'Ньютона':<15} | {res_n:<12.7f} | {it_n}")
print(f"{'Хорд':<15} | {res_s:<12.7f} | {it_s}")