import matplotlib.pyplot as plt
Dic={'Sleeping':8,'Classes':6,'Studying':3.5,'TV':2,'Music':1,'other':24-8-6-3.5-2-1} #Compile data into a dictionary
Sp= [0, 0, 0.1, 0, 0, 0,] #emphasis one activity
a=Dic.keys() #a is the keys of Dic
b=Dic.values() #b is values of Dic
plt.figure()
plt.pie(b, labels =a, startangle = 90, explode =Sp)  # create the par chart, and the "study" part will be moved out a little bit
plt.show() # show the image
plt.clf() #Clear Previous Image
activity=str(input("Input the activity you want to know (From 'sleep','classes','study','TV','music','other'):"))
print("You spent",Dic[activity],"hours on",activity,"during an average day.") # output the resule

