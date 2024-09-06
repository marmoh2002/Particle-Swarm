import numpy as np

def get_global_minima():
    return {
        'ackley': np.array([0.0, 0.0]),  
        'beale': np.array([3.0, 0.5]),   
        'booth': np.array([1.0, 3.0]),   
        'bukin': np.array([-10.0, 1.0]), 
        'cross_in_tray': np.array([1.349406, 1.349406]), 
        'drop_wave': np.array([0.0, 0.0]),  
        'easom': np.array([np.pi, np.pi]),        
        'eggholder': np.array([512.0, 404.2319]),  
        'goldstein_price': np.array([0.0, -2.0]), 
        'griewank': np.array([0.0, 0.0]),  
        'himmelblau': np.array([3.0, 2.0]),  
        'holder_table': np.array([8.055, 9.664]),  
        'langermann': np.array([0.0, 0.0]),  
        'levy': np.array([1.0, 1.0]),  
        'matyas': np.array([0.0, 0.0]), 
        'mccormick': np.array([-0.54719, -1.54719]),  
        'michalewicz': np.array([1.8013, 1.6013]),  
        'rastrigin': np.array([0.0, 0.0]),  
        'rosenbrock': np.array([1.0, 1.0]), 
        'schaffer_n2': np.array([0.0, 0.0]),  
        'schaffer_n4': np.array([0.0, 0.0]),  
        'schwefel': np.array([420.9687, 420.9687]),  
        'shubert': np.array([-1.9151, -1.9151]),  
        'sphere': np.array([0.0, 0.0]),  
        'styblinski_tang': np.array([-2.903534, -2.903534]),  
        'three_hump_camel': np.array([0.0, 0.0]),  
        'trid': np.array([1.0, 1.0])  
    }
