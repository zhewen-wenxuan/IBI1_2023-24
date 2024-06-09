import tensorflow as tf

# 创建一个张量
tensor = tf.constant([[1, 2], [3, 4]])

# 查看张量的属性
print("Tensor:", tensor)
print("Shape:", tensor.shape)
print("Data type:", tensor.dtype)

# 基本算术运算
tensor_add = tensor + 2
print("Tensor + 2:", tensor_add)

# 矩阵乘法
tensor_mul = tf.matmul(tensor, tensor)
print("Tensor matmul:", tensor_mul)

# 重塑张量
tensor_reshape = tf.reshape(tensor, [4])
print("Reshaped tensor:", tensor_reshape)
