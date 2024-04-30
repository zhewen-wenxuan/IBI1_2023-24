def DNA(dna):
    a=list(dna)
    b=len(dna)
    c=0
    for i in range (b):
        if a[i]=="C" or a[i]=="G":
            c=c+1
    return print(c*100/b,"%")
x=input('DNA sequence:')
DNA(x)