import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
os.chdir('/Users/xuanzhewen/code/IBI1_2023-24/Practical7')
dalys_data = pd.read_csv("dalys-rate-from-all-causes.csv")
a=dalys_data.loc[0:100:10,"DALYs"]
#输出a

dalys_AFG=dalys_data['Entity'] == 'Afghanistan'
b=dalys_data.loc[dalys_AFG,'DALYs']
#输出b

dalys_CHN=dalys_data['Entity']=='China'
china_data=dalys_data.loc[dalys_CHN, ['Year', 'DALYs']]
#输出china_data

dalys_list = china_data['DALYs'].tolist()
dalys_mean=np.mean(dalys_list)
c=china_data.loc[china_data['Year']==2019,'DALYs'].values[0]
#输出c
#输出dalys_mean
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

dalys_UK=dalys_data['Entity']=='United kingdom'
UK_data=dalys_data.loc[dalys_UK, ['Year', 'DALYs']]