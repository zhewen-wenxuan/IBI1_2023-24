import matplotlib.pyplot as plt
Dic={'Sleeping':8,'Classes':6,'Studying':3.5,'TV':2,'Music':1,'Other':3.5}
Sp= [0, 0, 0.1, 0, 0, 0,]
a=Dic.keys()
b=Dic.values()
plt.figure()
plt.pie(b, labels =a, startangle = 90, explode =Sp)
plt.show()
plt.clf() 