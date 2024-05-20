def name_Bond(x):
    y=x+18               # Calculate the age when the person turned 18.
    if 1973<=y<=1986:
        print("The actor is Roger Moore.")
    if 1987<=y<=1994:
        print('The actor is Timothy Dalton.')
    if 1995<=y<=2005:
        print('The actor is Pierce Brosnan.')
    if 2006<=y<=2021:
        print('The actor is Daniel Craig.')
    if (y<1973) or (y>2021):
        print('No actor according to the practical')
x=int(input("The year you born:"))
name_Bond(x)

