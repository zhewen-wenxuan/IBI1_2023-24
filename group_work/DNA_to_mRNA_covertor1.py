def DNA_mRNA_convertor(dna):
    a=list(dna)
    b=[]
    for i in range(len(dna)):
        if a[i]=="A":
            b.append("U")
        if a[i]=="T":
            b.append("A")
        if a[i]=="C":
            b.append("G")
        if a[i]=="G":
            b.append("C")
    return print("".join(b))
x='AGCTT'
DNA_mRNA_convertor (x)