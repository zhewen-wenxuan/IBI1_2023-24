import torch
import numpy as np
#
# #创建一个没有初始化的张量矩阵
# #创建一个指定形状的张量，但不初始化张量的元素。换句话说，张量中的数值是未定义的，是内存中原有数据的残留值。
x_tensor = torch.empty(5, 6)
# print('empty创建一个tensor', x_tensor)
#
# ##创建一个初始化的张量矩阵
# #创建一个指定形状的张量，并使用均匀分布的随机数进行初始化，随机数的范围是 [0, 1)。
# y_tensor = torch.rand(5, 6)
# print('rand(0,1)分布创建一个tensor', y_tensor)
#
# #创建一个全 0 tensor矩阵，并且设置类型不是int，而是long
# z_tensor = torch.zeros(5, 6, dtype=torch.long)
# print('创建一个全0矩阵', z_tensor)
#
# #new_ones 根据某一个张量构建新张量，但是类型是一致的，不过是全1形式
# c_tensor = x_tensor.new_ones(4,3, dtype = torch.double)
# print('new_ones result：', c_tensor)
#
# #rand_like进行张量复制，并且随机分配值
# d_tensor = torch.rand_like(x_tensor, dtype = torch.float)
# print('rand_like result：', d_tensor)
#
# #查看size
# print('查看x_tensor size:', x_tensor.size())
#
# #关于torch的加法
# e_tensor = torch.rand(5, 6)
# result = torch.add(y_tensor, e_tensor)
# print('add结果：', result)
#
# #另外一种加法
# other_result = y_tensor.add_(e_tensor)
# print('other add结果：', other_result)

#注意格式需要一致。
# f_tensor = torch.rand(5,4)
# new_result = torch.add(y_tensor, f_tensor)
# print('other add结果：', new_result)

# tensor changed to array
# g_tensor = torch.ones(5)
# print('tensor 格式内容：', g_tensor)
# print('格式转换结果：', g_tensor.numpy())

# array to tensor
# h = np.ones(5)
# print("转变前的格式：", h)
# print('转变后的格式', torch.from_numpy(h))

#关于cpu gpu转换
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     x_tensor = x_tensor.to(device)
#
# print(x_tensor)
# print(x_tensor.to('cpu', torch.double))


