import numpy as np
def rastrigin(x):
    """Rastrigin function"""
    return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

def sphere(x):
    """Sphere function"""
    return np.sum(x**2)

def rosenbrock(x):
    """Rosenbrock function"""
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))
def griewank(x):
    """Griewank function"""
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_part - prod_part

def trid(x):
    """Trid function"""
    n = len(x)
    sum_part = np.sum((x - 1)**2)
    product_part = np.sum(x[1:] * x[:-1])
    return sum_part - product_part
def ackley(x):
    """Ackley function"""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + 20 + np.e
def schwefel(x):
    """Schwefel function"""
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
