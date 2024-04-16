import numpy as np
import matplotlib.pyplot as plt

N = 10000  
Sus = 9999  
Inf = 10     
Rec = 0     
beta = 0.3  
gamma = 0.05  

susceptible_over_time = []  
infected_over_time = []    
recovered_over_time = []    

for i in range(1000):
    prob_infection = beta * Inf / N  
    prob_recovery = gamma            
    infected_indices = np.random.choice(range(N), size=int(prob_infection * Sus), replace=False)  
    recovered_indices = np.random.choice(range(N), size=int(prob_recovery * Inf), replace=False)  
    Sus -= len(infected_indices)  
    Inf += len(infected_indices) - len(recovered_indices)  
    Rec += len(recovered_indices)  

    # 将当前人数计数存储到相应的数组中
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
plt.savefig("SIR_model_plot.png", type="png")  # 保存图像
plt.show()
