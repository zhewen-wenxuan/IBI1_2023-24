uk_cities=[0.56,0.62,0.04,9.7]
ch_cities=[0.58,8.4,29.9,22.2]
uk_cityname=["Edinburgh","Glasgow","Stirling","London"]
china_cityname=["Haining","Hangzhou","Shanghai","Beijing"]
import matplotlib.pyplot as plt
width = 0.5
plt.figure()
plt.bar(uk_cityname,uk_cities, width)
plt.ylabel("Population")
plt.title("Population of UK cities")
plt.show()


plt.figure()
plt.bar(china_cityname,ch_cities, width)
plt.ylabel("Population")
plt.title("Population of China cities")
plt.show()
plt.clf()