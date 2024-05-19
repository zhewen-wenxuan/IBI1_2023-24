a=0.05 #add initial density
n=0 
while a<=0.9:
    n=n+1 #show the change of day
    a=a*2 #go each day,the density double
print('On day',str(n),'the cell density goes over 90%')
print('It takes',str(n-1),'days')
