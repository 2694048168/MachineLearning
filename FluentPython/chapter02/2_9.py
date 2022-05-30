#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 2-9 一个浮点类型数组的创建, 存入文件和从文件读取的过程
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-21
"""

import array
import random


# ----------------------------
if __name__ == "__main__":
    random.seed(42)
    # generator expression to create big-array
    float_array = array.array('d', (random.random() for i in range(10**7)))
    print(float_array[-1])

    file_ptr = open('floats.bin', 'wb')
    float_array.tofile(file_ptr)
    file_ptr.close()

    float_array_2 = array.array('d')
    file_ptr_2 = open('floats.bin', 'rb')
    float_array_2.fromfile(file_ptr_2, 10**7)
    file_ptr_2.close()
    print(float_array_2[-1])

    print(float_array_2[-1] == float_array[-1])
    # 这样利用二进制形式存储和读取浮点数, 高效
    # 另一个快速序列化数字类型的方法是使用 pickle 模块
    # https://docs.python.org/3/library/pickle.html

    # 通过改变数组中的一个字节来更新数组里的某一个元素的值
    # 利用 memoryview 和 struct 来操作二进制序列
    numbers_array = array.array('h', [-2, -1, 0, 1, 2])
    memory_view = memoryview(numbers_array)
    print(len(memory_view))
    print(memory_view[0])
    memory_view_oct = memory_view.cast('B')
    print(memory_view_oct.tolist())
    memory_view_oct[5] = 4
    print(numbers_array)