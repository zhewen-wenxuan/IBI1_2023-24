import numpy as np
import matplotlib.pyplot as plt

N = 10000  
Inf = 10  #If use Inf=1, the picture is not obvious
Rec = 0    
beta = 0.3  
gamma = 0.05  
vaccination_rates = [0.1, 0.2, 0.3, 0.4, 0.5,0.6]
infected_over_time ={vac: [] for vac in vaccination_rates}
time=1000
for vac in  vaccination_rates:
    Sus=N*(1-vac)-Rec-Inf
    for i in range(time):
        prob_inf = beta * Inf / N  
        prob_recovery = gamma            
        infected_indices = np.random.choice(range(N), size=int(prob_inf * Sus), replace=False)  
        recovered_indices = np.random.choice(range(N), size=int(prob_recovery * Inf), replace=False) 
        Sus -= len(infected_indices)   
        Inf += len(infected_indices) - len(recovered_indices)  
        Rec += len(recovered_indices)  
        # 将当前人数计数存储到相应的数组中
        infected_over_time[vac].append(Inf)

plt.figure(figsize=(8, 6), dpi=150)  
for vac_rate, infected_time in infected_over_time.items():
    plt.plot(time, infected_time, label=f'coverage={vac}')
plt.xlabel('time')
plt.ylabel('population')
plt.title('SIR Model')
plt.legend()
plt.savefig("SIR_model_plot.png", format="png")
plt.show()
