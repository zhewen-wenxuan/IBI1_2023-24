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
    result = "Both training regimes had the same improvement on running time."
print(result)