def calculator(total_money,price):
    number=total_money//price
    change=total_money%price
    return number,change

print(calculator(110,3))