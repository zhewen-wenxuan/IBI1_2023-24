def calculator(total_money,price):
    number=total_money//price
    change=total_money%price
    return number,change

total_money=float(input('The money you have ',))
price=float(input('the price of one bar ',))
print(calculator(total_money,price))