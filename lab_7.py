import math
import numpy as np

def func (x, func_number, deriviate):
    match (func_number):
        case "first":
            match (deriviate):
                case "bare":
                    return math.sin(x) + x**2
                case "sec":
                    return math.sin(x)*(2-x**2)+4*x*math.cos(x)
                case "fourth":
                    return (x**2-12)*math.sin(x)-8*math.cos(x)
                case _:
                    raise ValueError(f'Unknown param {deriviate}')
        case "second":
            match (deriviate):
                case "bare":
                    return math.exp(-x)*math.cos(x)
                case "sec":
                    return 2*math.exp(-x)*math.sin(x)
                case "fourth":
                    return -4*math.exp(-x)*math.cos(x)
                case _:
                    raise ValueError(f'Unknown param {deriviate}')
        case "third":
            match (deriviate):
                case "bare":
                    return 1/(1+x**2)
                case "sec":
                    return 8*(x**2)/(1+x**2)**3 - 2/(1+x**2)**2
                case "fourth":
                    return -288*(x**2)/(1+x**2)**4 + 24/(1+x**2)**3 + 384*(x**4)/(1+x**2)**5
                case _:
                    raise ValueError(f'Unknown param {deriviate}')
        case _:
            raise ValueError(f'Unknown func {func_number} ')








def trapez(a,b,n,  func_numb):
    if (b <= a): raise ValueError("b должно быть больше a")
    h = (b - a) / n
    I = 0
    error = abs(func(a,func_numb,"sec"))
    for i in range(n-1):
        I += func(a+i*h, func_numb, "bare") + func(a+h*(i+1), func_numb, "bare")
        error = max(error,abs(func(a+h*(i+1),func_numb,"sec")))
    I *= h/2
    error *= (h**2)*(b-a)/12
    return I,error

def simps(a,b,n, func_numb):
    if (b <= a): raise ValueError("b должно быть больше a")
    if (n % 2 != 0): raise ValueError("n должно быть чётным")

    h = (b - a) / n
    I = func(a,func_numb, "bare") + func(b,func_numb, "bare")
    error = max(abs(func(a,func_numb,"fourth")),abs(func(b,func_numb,"fourth")))  
    

    for i in range(1,n,2):
        I += 4 * func(a + h*i,func_numb, "bare")
        error = max(abs(func(a+h*i,func_numb,"fourth")), error)

    for i in range(2,n,2):
        I += 2 * func(a + h*i,func_numb, "bare")
        error = max(abs(func(a+h*i,func_numb,"fourth")), error)
    
    error *= (h**4)*(b-a)/180

    return (h / 3 ) * I, error


def compute(method,func_num,a,b,n_list:list):
    outV  = np.empty((3,0), float)

    for i in range(len(n_list)):
        current_n = n_list[i]    
        I , err =  method(a,b,current_n,func_num)
        newOut = np.array([[current_n],[I], [err]])
        outV = np.hstack((outV,newOut))
    print(outV)



a_1 = -5
b_1 = 5

a_2 = -4
b_2 = -2

a_3 = -4
b_3 = 4

n = list(range(50,1000,50))


print("First func start: sin(x)*x^2 ")
print("\t Trapez:")
compute(trapez,"first",a_1,b_1,n)

print("\tSimpson:")
compute(simps,"first",a_1,b_1,n)


print("Second func start: cos(x)*exp(-x) ")
print("\t Trapez:")
compute(trapez,"second",a_2,b_2,n)

print("\tSimpson:")
compute(simps,"third",a_2,b_2,n)



print("Third func start: 1/(1 + x^2) ")
print("\t Trapez:")
compute(trapez,"third",a_3,b_3,n)

print("\tSimpson:")
compute(simps,"third",a_3,b_3,n)





"""
def func_1(x ):
    return math.sin(x) + x**2

def func_1_der2(x):
    return math.sin(x)*(2-x**2)+4*x*math.cos(x)

def func_1_der4(x):
    return (x**2-12)*math.sin(x)-8*math.cos(x)



def func_2(x):
    return math.exp(-x)*math.cos(x)

def func_2_der2(x):
    return 2*math.exp(-x)*math.sin(x)

def func_2_der4(x):
    return -4*math.exp(-x)*math.cos(x)



def func_3(x):
    return 1/(1+x**2)

def func_3_der2(x):
    return 8*(x**2)/(1+x**2)**3 - 2/(1+x**2)**2

def func_3_der4(x):
    return -288*(x**2)/(1+x**2)**4 + 24/(1+x**2)**3 + 384*(x**4)/(1+x**2)**5

"""