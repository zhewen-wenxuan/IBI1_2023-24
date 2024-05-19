import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
os.chdir('/Users/xuanzhewen/code/IBI1_2023-24/Practical7')
dalys_data = pd.read_csv("dalys-rate-from-all-causes.csv")
a=dalys_data.loc[0:100:10,"DALYs"]
print(a)

dalys_AFG = []
for i in range(0,dalys_data.shape[0]):
    if dalys_data.iloc[i,0] == 'Afghanistan':
        dalys_AFG.append(True)
    else:
        dalys_AFG.append(False)  #ATTENTION: These lines select "Afghanistan" rows in "Entity" by using Boolean
print(dalys_data.loc[dalys_AFG,"DALYs"])   

dalys_CHN=dalys_data['Entity']=='China'
china_data=dalys_data.loc[dalys_CHN, ['Year', 'DALYs']]
print(china_data)

dalys_list = china_data['DALYs'].tolist()
dalys_mean=np.mean(dalys_list)
c=china_data.loc[china_data['Year']==2019,'DALYs'].values[0]
print(c)
print('The mean DALYs in China is',dalys_mean)
if dalys_mean > c:
    print("The DALYs in China in 2019 was less than the mean.")
elif dalys_mean < c:
    print("The DALYs in China in 2019 was larger than the mean.")
else:
    print("The DALYs in China in 2019 was equal to the mean.")

plt.plot(china_data.Year, china_data.DALYs, 'b+')
plt.xlabel('Year')
plt.ylabel('DALYs')
plt.title('DALYs in China with Year')
plt.xticks(china_data.Year,rotation=-90)
plt.show()

dalys_UK=dalys_data['Entity']=='United Kingdom'
UK_data=dalys_data.loc[dalys_UK, ['Year', 'DALYs']]
plt.plot(china_data.Year, UK_data.DALYs, 'r--', label="UK")
plt.plot(china_data.Year, china_data.DALYs, 'b-', label="China")
plt.xlabel('Year')
plt.ylabel('DALYs')
plt.title('Comparation between UK and China')
plt.xticks(UK_data.Year,rotation=-90)
plt.legend()
plt.show()

