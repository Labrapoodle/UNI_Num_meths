import numpy as np

def solve_system(A, b, epsilon=0.001, max_iter=100):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    # Проверка
    for i in range(n):
        if abs(A[i, i]) <= np.sum(np.abs(A[i, :])) - abs(A[i, i]):
            print(f"Внимание: Строка {i} не имеет строгого диагонального преобладания.")

    #  МПИ
    def simple_iteration():
        x = np.zeros(n)
        for it in range(1, max_iter + 1):
            x_new = np.zeros(n)
            for i in range(n):
                s = sum(A[i, j] * x[j] for j in range(n) if i != j)
                x_new[i] = (b[i] - s) / A[i, i]
            
            if np.linalg.norm(x_new - x, ord=np.inf) < epsilon:
                return x_new, it
            x = x_new
        return None, max_iter

    # 2. Зейдель
    def seidel_method():
        x = np.zeros(n)
        for it in range(1, max_iter + 1):
            x_old = np.copy(x)
            for i in range(n):
                
                s1 = sum(A[i, j] * x[j] for j in range(i))      # Новые значения
                s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n)) # Старые значения
                x[i] = (b[i] - s1 - s2) / A[i, i]
            
            # Проверка сходимости
            if np.linalg.norm(x - x_old, ord=np.inf) < epsilon:
                return x, it
        return None, max_iter

    
    sol_simple, iter_simple = simple_iteration()
    sol_seidel, iter_seidel = seidel_method()

    print("="*40)
    if sol_seidel is not None:
        print("РЕШЕНИЕ СИСТЕМЫ (Метод Зейделя):")
        for i, val in enumerate(sol_seidel):
            print(f"x[{i}] = {val:.6f}")
    
    print("-" * 40)
    print(f"ИТОГ СРАВНЕНИЯ (epsilon={epsilon}):")
    print(f"Количество итераций метода простых итераций = {iter_simple}")
    print(f"Количество итераций метода Зейделя           = {iter_seidel}")
    print("="*40)

matrix_A = [[10, 1, 1], 
            [2, 10, 1], 
            [2, 2, 10]]
vector_b = [12, 13, 14]

solve_system(matrix_A, vector_b)