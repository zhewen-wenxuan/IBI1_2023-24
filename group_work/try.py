list=input()
t=0
for i in range(0,len(list)):
    b=list[i]
    if b=='C'or b=='G':
        t=t+1
q=t/len(list)
print(t,q)