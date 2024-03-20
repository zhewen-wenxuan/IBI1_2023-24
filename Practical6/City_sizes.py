uk_cities=[0.56,0.62,0.04,9.7]
ch_cities=[0.58,8.4,29.9,22.2]
import numpy as np
import matplotlib.pyplot as plt
N=4
std_err = [2, 1, 2, 3]
ind = np.arange(N)
width = 0.5
plt.figure()
plt.bar(ind,uk_cities, width, yerr=std_err)
plt.ylabel("Population")
plt.title("Population of UK cities")
plt.xticks(ind)
plt.show()


plt.figure()
plt.bar(ind,ch_cities, width, yerr=std_err)
plt.ylabel("Population")
plt.title("Population of China cities")
plt.xticks(ind)
plt.show()
plt.clf()