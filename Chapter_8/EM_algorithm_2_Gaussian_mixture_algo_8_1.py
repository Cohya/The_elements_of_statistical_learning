import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt 
import sys 

def calculate_log_likelihood(Y, gamma_list , phi_theta_1, phi_theta_2, pi,  n):
    
   
    for i in range(n):
        gamma_i = gamma_list[i]
        
        a = ((1- gamma_i) * np.log(phi_theta_1.pdf(Y[i])) 
            + gamma_i*np.log(phi_theta_2.pdf(Y[i])))
        
        b = ((1 - gamma_i) * np.log(1-pi)
             +  gamma_i * np.log(pi))
    
    log_likelihood = a + b
    
    return log_likelihood 
        
    
def gaussian_2_mixture(Y, mu_init = [], var_init = [], pi_init = 0.5, verbose = True):
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
    
    
    max_iteration  = 1000
    # Expectation Step 
    gamma_list = [0 for _ in range(n)]
    
    ## loop until convergence
    iteration = 0
    error = 1
    log_likelihood_vec = [-sys.maxsize]
    while error > 1e-15 and iteration < max_iteration and iteration < 100:
   
        # Expectation Step
        phi_theta_2 = norm(loc = mu_2, scale = np.sqrt(var_2))
        phi_theta_1 = norm(loc = mu_1, scale = np.sqrt(var_1))
        
        for i in range(n):
            g_i = (pi * phi_theta_2.pdf(Y[i])/((1 - pi)*phi_theta_1.pdf(Y[i])
                   + pi * phi_theta_2.pdf(Y[i])))
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

        log_likelihood = calculate_log_likelihood(Y, gamma_list , phi_theta_1,
                                                  phi_theta_2, pi,  n)
        
        
        iteration += 1
        error = np.abs(log_likelihood_vec[-1] - log_likelihood)
        log_likelihood_vec.append(log_likelihood)
        
        if iteration % 10 ==0 :
            print("iteration: %i, loglikelihood: %.5f, error: %.6f, pi: %.2f" % 
                  (iteration,log_likelihood, error, pi))
        
    if verbose:
        plt.plot(log_likelihood_vec[1:],'-*')
        plt.ylabel('log-likelihood')
        plt.xlabel('Iteration')
    return mu_1, var_1, mu_2, var_2, pi
    
def main():
    Y = [-0.39, 0.12, 0.94, 1.67, 1.76, 2.44,
         3.72, 4.28, 4.92, 5.53, 0.06, 0.48,
         1.01, 1.68, 1.80, 3.25, 4.12, 4.60,
         5.28, 6.22]
    
    mu_1, var_1, mu_2, var_2, pi =   gaussian_2_mixture(Y)
    
    print("mu_1: %.2f, var_1: %.2f, mu_2: %.2f, var_2: %.2f, pi: %.2f" %
          (mu_1, var_1, mu_2, var_2, pi))
    
if __name__ == "__main__":
    main()    
    