import torch

# 创建一个张量
tensor = torch.tensor([[1, 2], [3, 4]])

# 查看张量的属性
print("Tensor:", tensor)
print("Shape:", tensor.shape)
print("Data type:", tensor.dtype)

# 基本算术运算
tensor_add = tensor + 2
print("Tensor + 2:", tensor_add)

# 矩阵乘法
tensor_mul = torch.matmul(tensor, tensor)
print("Tensor matmul:", tensor_mul)

# 重塑张量
# 关于view，view 方法要求张量在内存中是连续的。如果张量不是连续的，可以使用 tensor.contiguous() 方法将其转为连续后再使用 view。
'''
tensor 原本是一个 2x2 的矩阵（二维张量），通过 tensor.view(4) 将其重塑为一个包含 4 个元素的一维张量。
'''
tensor_reshape = tensor.view(4)
print("Reshaped tensor:", tensor_reshape)
