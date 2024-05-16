def Complementary_DNA(dna):
    a=list(dna)
    b=[]
    c=[]
    for i in range(len(a)):
        if a[i]=="A":
            b.append("T")
        if a[i]=="T":
            b.append("A")
        if a[i]=="C":
            b.append("G")
        if a[i]=="G":
            b.append("C")
    for i in range(len(a)):
        c.append(b[len(a)-1-i])
    return print("".join(c))
x='AGCTT'
Complementary_DNA(x)