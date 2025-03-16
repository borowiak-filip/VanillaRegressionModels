import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

lb = 1 # lowerbound
ub = 20 # upperbound

counts = np.zeros(ub) # init counts 

def mcmc(epochs):
    
    # init sampling x0
    x0 = np.random.randint(lb, ub+1)
    
    for _ in range(epochs):
        
        # proposed distribution randomly chnages value +- 1
        alpha = np.random.randint(0, 2)
        
        if alpha == 1:
            w_prop = x0 - 1
        else:
            w_prop = x0 + 1
        
        # calculate a, reject if not in bound
        r = range(lb, ub+1)
        if w_prop in r:
            x0 = w_prop
            
        counts[x0-1] += 1

    
    return counts
    

counts = mcmc(100000)

plt.bar(range(lb, ub+1), counts)
plt.title("MCMC results")
plt.show()