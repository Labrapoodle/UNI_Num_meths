import numpy as np

MAX_ORDER = 3
EPS=1e-12

def solve_gauss(matrix_A, vector_b, eps=1e-12):
    
    
    A = matrix_A.astype(float)
    b = vector_b.astype(float).reshape(-1, 1)
    n = len(b)
    
    
    system = np.hstack((A, b))

    # Прямой ход 
    for i in range(n):
        
        pivot = np.abs(system[i:, i]).argmax() + i
        pivot_val = system[pivot, i]

        if np.abs(pivot_val) < eps:
            raise ValueError(f"Матрица вырождена: столбец {i} ниже строки {i} содержит только нули.")

        
        system[[i, pivot]] = system[[pivot, i]]
        
        
        system[i] = system[i] / system[i, i]

        for j in range(i + 1, n):
            k = system[j, i] 
            system[j] = system[j] - k * system[i]

    #  Обратный ход 
    for i in range(n - 1, 0, -1):
        for j in range(0, i):
            k = system[j, i]
            system[j] = system[j] - k * system[i]

    
    return system[:, -1]


mx = np.random.randint(-3, 3, size=(MAX_ORDER,MAX_ORDER)).astype(float)
f = np.random.randint(-3, 3, size=(MAX_ORDER,1)).astype(float)

result = solve_gauss(mx,f)
print(result)

