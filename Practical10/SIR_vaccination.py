import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('MacOSX')

N = 10000  
Inf = 1  #If use Inf=1, the picture is not obvious
Rec = 0    
beta = 0.3  
gamma = 0.05  
vaccination_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
infected_over_time ={vac: [] for vac in vaccination_rates}
time=1000
for vac in  vaccination_rates:
    Inf=1
    Rec=0
    Sus=int(N*(1-vac))-Rec-Inf
    for i in range(time):
        prob_inf = beta * Inf / N  
        prob_recovery = gamma            
        infected_indices = np.random.binomial(Sus, prob_inf)
        recovered_indices = np.random.binomial(Inf, prob_recovery)
        Sus -= infected_indices   
        Inf += infected_indices - recovered_indices  
        Rec += recovered_indices  
        # Store the current number of people count in the corresponding array
        infected_over_time[vac].append(Inf)

plt.figure(figsize=(8, 6), dpi=150)  
for vac_rate, infected_time in infected_over_time.items():
    plt.plot(range(time), infected_time, label=f'coverage={vac_rate}')
plt.xlabel('time')
plt.ylabel('population')
plt.title('SIR Model')
plt.legend()
plt.show()
