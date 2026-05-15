import numpy as np
import matplotlib.pyplot as plt


x_start = 0.0
x_end = 1.0
y_start = 0.0
y_end_target = 0.5 * np.exp(1)  
n_steps = 100
h = (x_end - x_start) / n_steps


def f(x, state):
    y, z = state
    dy = z
    dz = 2 * z - y + np.exp(x)
    return np.array([dy, dz])


def rk4_step(x, state, h):
    k1 = f(x, state)
    k2 = f(x + h/2, state + h/2 * k1)
    k3 = f(x + h/2, state + h/2 * k2)
    k4 = f(x + h, state + h * k3)
    return state + (h/6) * (k1 + 2*k2 + 2*k3 + k4)


def solve_cauchy(s):
    x_values = np.linspace(x_start, x_end, n_steps + 1)
   
    state = np.array([y_start, s])
    
    results = [state[0]]
    curr_state = state
    
    for i in range(n_steps):
        curr_state = rk4_step(x_values[i], curr_state, h)
        results.append(curr_state[0])
        
    return x_values, np.array(results), results[-1]


s1, s2 = 0.0, 1.0
_, _, y1_final = solve_cauchy(s1)
_, _, y2_final = solve_cauchy(s2)


f1 = y1_final - y_end_target
f2 = y2_final - y_end_target


if abs(f2 - f1) < 1e-12:
    print("Ошибка: невязки одинаковы, метод стрельбы не применим.")
else:
    
    s_ideal = s2 - f2 * (s2 - s1) / (f2 - f1)
    print(f"Подобранный начальный наклон s: {s_ideal:.6f}")

    
    x_final, y_final, target_check = solve_cauchy(s_ideal)
    print(f"Значение на правой границе: {target_check:.6f}")
    print(f"Целевое значение (0.5*e): {y_end_target:.6f}")

   
    plt.figure(figsize=(10, 6))
    plt.plot(x_final, y_final, 'b-', label='Численное решение (РК4)')
    plt.plot(x_end, y_end_target, 'ro', label='Целевая точка (1, 0.5e)')
    plt.title('Решение краевой задачи методом стрельбы')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()