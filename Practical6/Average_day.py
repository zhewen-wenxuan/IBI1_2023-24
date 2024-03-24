import matplotlib.pyplot as plt
e=input()
Dic={'Sleeping':8,'Classes':6,'Studying':3.5,'TV':2,'Music':1,'strength training100':e} #Compile data into a dictionary
Sp= [0, 0, 0.1, 0, 0, 0,] #emphasis one activity
a=Dic.keys() #a is the keys of Dic
b=Dic.values() #b is values of Dic
plt.figure()
plt.pie(b, labels =a, startangle = 90, explode =Sp)
plt.show()
plt.clf() #Clear Previous Image