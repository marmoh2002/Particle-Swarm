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
def levy(x):
    """Levy function"""
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3
def michalewicz(x):
    """Michalewicz function"""
    m = 10  # steepness factor
    d = len(x)
    i = np.arange(1, d + 1)
    return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi)**(2 * m))
def easom(x):
    """Easom function"""
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))

def beale(x):
    """Beale function"""
    x1, x2 = x
    term1 = (1.5 - x1 + x1*x2)**2
    term2 = (2.25 - x1 + x1*x2**2)**2
    term3 = (2.625 - x1 + x1*x2**3)**2
    return term1 + term2 + term3

def goldstein_price(x):
    """Goldstein-Price function"""
    x1, x2 = x
    term1 = (1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))
    term2 = (30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
    return term1 * term2

