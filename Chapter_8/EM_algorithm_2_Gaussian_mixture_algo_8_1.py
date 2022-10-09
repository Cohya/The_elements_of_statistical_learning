import numpy as np 
from scipy.stats import norm

def gaussian_2_mixture(Y, mu_init = [], var_init = [], pi_init = 0.5):
    """
    

    Parameters
    ----------
    Y : list
        data.
    mu_init : list, optional
        initial guesses for the average values. The default is [].
    var_init : list, optional
        initial guesses for the variances. The default is [].
    pi_init : float from [0,1], optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    None.

    """
    # Initialization step 
    if len(var_init) == 0 :
        var_1 = var_2 = np.std(Y) ** 2 
    else:
        var_1, var_2 = var_init
        
    if len(mu_init) == 0 :
        mu_1 = np.random.choice(Y, size = 1)
        mu_2 = np.random.choice(Y, size = 1)
    else:
        mu_1, mu_2 = mu_init
    
    pi = pi_init
    
    n = len(Y)
    
    
    # Expectation Step 
    gamma_list = [0 for _ in range(n)]
    phi_theta_2 = norm(loc = mu_2, scale = np.sqrt(var_2))
    phi_theta_1 = norm(loc = mu_1, sclae = np.sqrt(var_1))
    for i in range(n):
        g_i = pi * phi_theta_2(Y[i])/((1 - pi)*phi_theta_1(Y[i]) + pi * phi_theta_2(Y[i]))
        gamma_list[i] = g_i
        
    # Maximization Step 
    a = 0
    b = 0 
    c = 0 
    d = 0
    for i in range(n):
        gamma_i = gamma_list[i]
        a += gamma_i * Y[i]
        b += gamma_i
        c += (1 - gamma_i) * Y[i]
        d += (1 - gamma_i)
    
    mu_2 = a / b 
    mu_1 = c / d
    
    e = 0
    f = 0
    for i in range(n):
        gamma_i = gamma_list[i]
        e += gamma_i * (Y[i] - mu_2)**2
        f += (1 - gamma_i) * (Y[i] - mu_1)**2
    
    var_1 = f / d
    var_2 = e / b 

    pi = b / n

             
           
    
    
        
        
            
        
    
    