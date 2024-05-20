import numpy as np
import matplotlib.pyplot as plt

N = 10000  
Sus = 9999
Inf = 1  # If use Inf=1, the picture is not obvious
Rec = 0     
beta = 0.3  
gamma = 0.05  

susceptible_over_time = []  
infected_over_time = []    
recovered_over_time = []    

for i in range(1000):
    prob_infection = beta * Inf / N  
    prob_recovery = gamma            
    
    new_infections = np.random.binomial(Sus, prob_infection)
    new_recoveries = np.random.binomial(Inf, prob_recovery)
    
    Sus -= new_infections  
    Inf += new_infections - new_recoveries
    Rec += new_recoveries  

    susceptible_over_time.append(Sus)
    infected_over_time.append(Inf)
    recovered_over_time.append(Rec)

plt.figure(figsize=(8, 6), dpi=150)  
plt.plot(susceptible_over_time, label='susceptible')
plt.plot(infected_over_time, label='infected')
plt.plot(recovered_over_time, label='recovered')
plt.xlabel('time')
plt.ylabel('population')
plt.title('SIR Model')
plt.legend()
plt.show()
