import numpy as np

def simple_iteration_method(A, b, epsilon, max_iter):
    # ШАГ 0: Подготовка данных
    A = np.array(A, dtype=float) 
    b = np.array(b, dtype=float) 
    n = len(b) 
    
    # ШАГ 1: Проверка на диагональное преобладание
    for i in range(n): 
        diagonal = abs(A[i, i]) 
        row_sum = np.sum(np.abs(A[i, :])) - diagonal 
        if diagonal <= row_sum: 
            print(f"Предупреждение: Нет диагонального преобладания в строке {i}") 
            print("Метод может не сойтись. Требуется преобразование системы.") 

    # ШАГ 2: Преобразование к виду x = Bx + d
    B = np.zeros((n, n)) 
    d = np.zeros(n) 
    
    for i in range(n): 
        if A[i, i] == 0: 
            raise ValueError("Нулевой элемент на диагонали. Метод неприменим.") 
        
        d[i] = b[i] / A[i, i] 
        for j in range(n): 
            if i == j: 
                B[i, j] = 0 
            else: 
                B[i, j] = -A[i, j] / A[i, i] 

    # Проверка нормы матрицы B
    norm_B = np.max(np.sum(np.abs(B), axis=1)) 
    print(f"Норма матрицы итераций ||B|| = {norm_B}") 
    if norm_B >= 1: 
        print("Условие сходимости ||B|| < 1 не выполняется.") 

    # ШАГ 3: Инициализация
    x_old = np.zeros(n) 
    converged = False 
    iter_count = 0 

    # ШАГ 4: Итерационный процесс
    while iter_count < max_iter and not converged: 
        iter_count += 1 
        
        # x_new = B * x_old + d
        x_new = np.dot(B, x_old) + d 
        
        # ШАГ 5: Оценка погрешности
        error = np.max(np.abs(x_new - x_old)) 
        print(f"Итерация {iter_count}, Погрешность: {error}") 
        
        if error < epsilon: 
            converged = True 
        else: 
            x_old = np.copy(x_new) 

    # ШАГ 6: Вывод результатов
    if converged: 
        print(f"\nРешение найдено за {iter_count} итераций:") 
        for i in range(n): 
            print(f"x[{i}] = {x_new[i]}") 
        return x_new
    else: 
        print(f"\nМетод не сошёлся за {max_iter} итераций.") 
        return None

# Пример использования:
matrix_A = [[10, 1, 1], 
            [2, 10, 1], 
            [2, 2, 10]]
vector_b = [12, 13, 14]

solution = simple_iteration_method(matrix_A, vector_b, epsilon=0.001, max_iter=100)