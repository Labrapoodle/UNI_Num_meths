import numpy as np
import matplotlib.pyplot as plt

def golden_section_search(f, a, b, eps, find_max=False):
    
    phi = (1 + 5**0.5) / 2
    
    
    
    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi
    
    y1 = f(x1)
    y2 = f(x2)
    
    while abs(b - a) > eps:
        if find_max:
            # для максимума
            condition = y1 <= y2
        else:
            # для минимума
            condition = y1 >= y2
            
        if condition:
            a = x1
            x1 = x2
            y1 = y2
            x2 = a + (b - a) / phi
            y2 = f(x2)
        else:
            b = x2
            x2 = x1
            y2 = y1
            x1 = b - (b - a) / phi
            y1 = f(x1)
            
    
    x_extrimum = (a + b) / 2
    return x_extrimum


def f(x):
    return (x - 2)**2 + 3  


a, b = -2, 6
epsilon = 1e-5
is_max = False # ищем минимум


result_x = golden_section_search(f, a, b, epsilon, find_max=is_max)
result_y = f(result_x)

print(f"Экстремум найден в точке x = {result_x:.6f}")
print(f"Значение функции f(x) = {result_y:.6f}")


x_vals = np.linspace(a - 1, b + 1, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='f(x)', color='blue')
plt.axvline(a, color='gray', linestyle='--', label='Границы [a, b]')
plt.axvline(b, color='gray', linestyle='--')


plt.scatter(result_x, result_y, color='red', s=100, zorder=5, 
            label=f'Extremum: ({result_x:.4f}, {result_y:.4f})')

plt.title(f"Метод золотого сечения (поиск {'максимума' if is_max else 'минимума'})")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()