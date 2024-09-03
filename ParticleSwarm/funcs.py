import numpy as np
import inspect

def ackley(x):
    """Ackley function"""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + 20 + np.e

def beale(x):
    """Beale function"""
    x1, x2 = x
    term1 = (1.5 - x1 + x1*x2)**2
    term2 = (2.25 - x1 + x1*x2**2)**2
    term3 = (2.625 - x1 + x1*x2**3)**2
    return term1 + term2 + term3

def booth(x):
    """Booth function"""
    x1, x2 = x
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

def bukin(x):
    """Bukin function N.6"""
    x1, x2 = x
    return 100 * np.sqrt(np.abs(x2 - 0.01*x1**2)) + 0.01*np.abs(x1 + 10)

def cross_in_tray(x):
    """Cross-in-tray function"""
    x1, x2 = x
    return -0.0001 * (np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(100 - np.sqrt(x1**2 + x2**2)/np.pi))) + 1)**0.1

def drop_wave(x):
    """Drop-Wave function"""
    x1, x2 = x
    frac1 = 1 + np.cos(12 * np.sqrt(x1**2 + x2**2))
    frac2 = 0.5 * (x1**2 + x2**2) + 2
    return -frac1 / frac2

def easom(x):
    """Easom function"""
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))

def eggholder(x):
    """Eggholder function"""
    x1, x2 = x
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1/2 + 47))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

def goldstein_price(x):
    """Goldstein-Price function"""
    x1, x2 = x
    term1 = (1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))
    term2 = (30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
    return term1 * term2

def griewank(x):
    """Griewank function"""
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_part - prod_part

def himmelblau(x):
    """Himmelblau's function"""
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def holder_table(x):
    """Holder table function"""
    x1, x2 = x
    return -np.abs(np.sin(x1) * np.cos(x2) * np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2)/np.pi)))

def langermann(x):
    """Langermann function"""
    x1, x2 = x
    A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
    c = np.array([1, 2, 5, 2, 3])
    m = 5
    return np.sum(c * np.exp(-(1/np.pi) * ((x1 - A[:, 0])**2 + (x2 - A[:, 1])**2)) * np.cos(np.pi * ((x1 - A[:, 0])**2 + (x2 - A[:, 1])**2)))

def levy(x):
    """Levy function"""
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def matyas(x):
    """Matyas function"""
    x1, x2 = x
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2

def michalewicz(x):
    """Michalewicz function"""
    m = 10  # steepness factor
    d = len(x)
    i = np.arange(1, d + 1)
    return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi)**(2 * m))

def mccormick(x):
    """McCormick function"""
    x1, x2 = x
    return np.sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1

def rastrigin(x):
    """Rastrigin function"""
    return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

def rosenbrock(x):
    """Rosenbrock function"""
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

def schaffer_n2(x):
    """Schaffer N.2 function"""
    x1, x2 = x
    return 0.5 + (np.sin(x1**2 - x2**2)**2 - 0.5) / (1 + 0.001*(x1**2 + x2**2))**2

def schaffer_n4(x):
    """Schaffer N.4 function"""
    x1, x2 = x
    return 0.5 + (np.cos(np.sin(np.abs(x1**2 - x2**2)))**2 - 0.5) / (1 + 0.001*(x1**2 + x2**2))**2

def schwefel(x):
    """Schwefel function"""
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def shubert(x):
    """Shubert function"""
    x1, x2 = x
    sum1 = sum(i * np.cos((i+1)*x1 + i) for i in range(1, 6))
    sum2 = sum(i * np.cos((i+1)*x2 + i) for i in range(1, 6))
    return sum1 * sum2

def sphere(x):
    """Sphere function"""
    return np.sum(x**2)


def styblinski_tang(x):
    """Styblinski-Tang function"""
    return 0.5 * np.sum(x**4 - 16*x**2 + 5*x)

def three_hump_camel(x):
    """Three-hump camel function"""
    x1, x2 = x
    return 2*x1**2 - 1.05*x1**4 + x1**6/6 + x1*x2 + x2**2
 
def trid(x):
    """Trid function"""
    n = len(x)
    sum_part = np.sum((x - 1)**2)
    product_part = np.sum(x[1:] * x[:-1])
    return sum_part - product_part

def parse_user_function(func_str):
    """Parse user input string into a callable function."""
    try:
        return lambda x: eval(func_str, {"x": x, "np": np})
    except:
        print("Invalid function. Please try again.")
        return None