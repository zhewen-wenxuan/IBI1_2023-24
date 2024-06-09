#case1_关于Python的编码 utf-8 以及写入jsonl文件中 ensure_ascii=False
# def save_to_jsonl(file_path, data):
#     with open(file_path, 'a', encoding='utf-8') as f:
#         json_line = json.dumps(data, ensure_ascii=False)
#         f.write(json_line + '\n')
#
#case2 关于print和 return *****************************************************************
# a = 10
# b = 5
# c = 3
# def add(a,b):
#     sum = a + b
#     print(sum)
#     # print("两个运算和为：" + sum)
#     # print("两个运算和为：" + str(sum))
#     # print("两个运算和为：", sum)
#
#     return sum
# #
# def multi_process(a, b, c):
#     sum = add(a, b)
#     print('sum:' + str(sum))
#     result = sum / c
#     print(result)
#     print(sum / c)
#
# #执行
# add(a,b)
# multi_process(a,b,c)

# #case3关于输入input和传参**************************************************
# name = input('请输入您的姓名:')
# print(name)


#case4 多行注释  ******************************************
'''
这是多行注释；
日期2024.5.19
欢迎使用python
23123
56+5656
56454655
56454
'''
# import random

#case5 关于缩进 ***************************
# def TAB():
#     tag = 'just case'
#     print(tag)

#case6 保留字show  ***************************
# import keyword
# print(keyword.kwlist)
# print(len(keyword.kwlist))

#case7 布尔值
# a = 10
# b = 5
# if a > b:
#     tag = True
# else:
#     tag = False
# print(tag)

#case8 eval()and input()
# text = '3+5'
# print(text, type(text))
# eval_text = eval(text)
# print(eval_text, type(eval_text))

# age = eval(input('Please enter your age: '))
# age = input('Please enter your age: ')
#
# print("Your age is ", age, type(age))


#case9 关于= 和 == 的区别
# a = 15
# b = 5
# if a / b == 3:
# # if a / b = 3:
#    print("a是b的三倍")
# else:
#     pass

#case10 关于逻辑 与或非 运算符
# a = 15
# b = 3
# c = 5

# if a / b == 5 and a < c:
#     print("a是b的三倍，并且c小于a")
# else:
#     print("123")
#
# if a > b or b > c:
#     print('要么a比b大，要么b比c大，或者全部成立')
# else:
#     pass
#
# if a not in range(1, 10):
#     print('a不在range范围内')
# else:
#     print('a在range范围内')


#case11 程序编写——顺序结构 case11
# a = 1
# b = 5
# c = 2
# result = a * b * c
# print('连乘结果是', result)
# if result % 2 == 0:
#     print('三者之和是偶数')
# else:
#     print('三者之和是奇数')


#case12 关于if条件结构 判断学号是否是8位
# id = input('请输入学号（8位）')
# if len(id) < 8:
#     print('您输入的学号位数少于8位')
# elif len(id) == 8:
#     print('学号有效')
# elif len(id) > 8:
#     print('您输入的学号位数大于8位')

# id = input('请输入学号（8位）')
# if len(id) < 8:
#     print('您输入的学号位数少于8位')
# elif len(id) == 8:
#     print('学号有效')
# else:
#     print('您输入的学号位数大于8位')


#case13 if嵌套 判断学号是否是8位，并且其中包括x
# id = input('请输入学号（8位）')
# if len(id) < 8:
#     print('您输入的学号位数少于8位')
#
# elif len(id) > 8:
#     print('您输入的学号位数大于8位')
#
# elif len(id) == 8:
#     if 'x' in id:
#         print('学号有效')
#     else:
#         print('学号长度符合8位，但是没有包括x')
#
# elif len(id) == 8 and 'x' in id:


#case14 if多条件判断 判断学号是否是8位，并且其中包括x
# id = input('请输入学号（8位）')
#
# if len(id) == 8 and 'x' in id:
#     print('学号有效')
# else:
#     print('学号无效')

#case15 关于for循环
# x = 0
# for k in range(1,5): #从0-4
#     x += k
# print(x)
#
# y = ''
# for i in ('a', 'b', 'c'):
#     y += i
# print(y)

#case16 while循环
# i = 5
# while i > 0:
#     print('这是第', i, '轮')
#     i -= 1
# else:
#     print('此时i=0')

#case17 利用while模拟用户输入信息情况。用户可以输入最多5次，如果还是失败，那么用户将无法再次输入信息，只能等待时间。
# i = 5
# while i > 0:
#     input_text = input('请输入学号信息')
#     if len(input_text) != 8:
#         print('学号错误，请重新输入')
#     else:
#         print('True', input_text)
#         break
#     i -= 1
# else:
#     print('请稍后重试')

#case18 利用while写一个无限循环的用户登录，用户输入用户名，密码，然后实现的登录成功
# import random
# tag = True
# i = 1
# if tag:
#     while i > 0:
#         print('欢迎使用AI教学平台，请输入您的用户名和密码')
#         user_name = input('请输入用户名')
#         password = input('请输入密码')
#         code = str(random.randint(1000, 9999))
#         valid_code = input('请输入' + code)
#         if str(valid_code) == code:
#             print('登录成功！')
#             print('您的用户名是：', user_name, '，您的密码是：', password)
#             print('****************************************************')
#         else:
#             print('登录失败！')
#             print('****************************************************')
#         i += 1
#
#     else:
#         pass
# else:
#     pass



#case19 利用嵌套循环写一个4*8的长方形，4*4的正方形
# def create_pattern(long, wide):
#     for i in range(long):
#         part = ''
#         for k in range(wide):
#             part += '*'
#         print(part)
#
# create_pattern(4, 8)
# create_pattern(4, 4)
#产生三角形
# part = ''
# for i in range(1, 5):
#     part = '*' * i
#     print(part)


# case20 break
# import random
# tag = True
# i = 1
# count = 0
# if tag:
#     while i > 0:
#         print('欢迎使用AI教学平台，请输入您的用户名和密码')
#         user_name = input('请输入用户名')
#         password = input('请输入密码')
#         code = str(random.randint(1000, 9999))
#         valid_code = input('请输入' + code)
#         if str(valid_code) == code:
#             print('登录成功！')
#             print('您的用户名是：', user_name, '，您的密码是：', password)
#             print('****************************************************')
#         else:
#             print('登录失败！')
#             print('****************************************************')
#         i += 1
#         count += 1
#         if count > 100:
#             break
#
#     else:
#         pass
# else:
#     pass


#case21 continue
# import random
# tag = True
# i = 1
# count = 0
# if tag:
#     while i > 0:
#         if count == 2:
#             print('跳过了第三次操作')
#             print('****************************************************')
#             count += 1
#             i += 1
#             continue
#         else:
#             print('欢迎使用AI教学平台，请输入您的用户名和密码')
#             user_name = input('请输入用户名')
#             password = input('请输入密码')
#             code = str(random.randint(1000, 9999))
#             valid_code = input('请输入' + code)
#             if str(valid_code) == code:
#                 print('登录成功！')
#                 print('第', i, '，您的用户名是：', user_name, '，您的密码是：', password)
#                 print('****************************************************')
#             else:
#                 print('登录失败！')
#                 print('****************************************************')
#             i += 1
#             count += 1
#
#     else:
#         pass
# else:
#     pass






