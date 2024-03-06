a=40
b=36
c=30
d=a-b
e=b-c
if d > e:
    result = "Running only had a greater improvement on running time."
elif e > d:
    result = "Running and strength training had a greater improvement on running time."
else:
    result = "They are same."
print(result)


X=True
Y=False
W=X!=Y
print("X:", X)
print("Y:", Y)
print("W:", W)

# X Y W
# T T F
# T F T
# F T T
# F F F