def favorite_bond(birth_year):
    actors = {"Roger Moore": 1973,"Timothy Dalton": 1987,"Pierce Brosnan": 1995,"Daniel Craig": 2006}
    for actor,year in actors.items():
        if birth_year >=year:
            return actor
        else:
            return "None"
        
b=1988
print(favorite_bond(b))

a=int(input('Your birth'))
print(favorite_bond(a)) 