#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 2-10 NumPy and SciPy 高效处理和计算数值
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-21

如何最大限度地利用 Python 标准库,但是 NumPy 和 SciPy 的优秀,
凭借着 NumPy 和 SciPy 提供的高阶数组和矩阵操作， Python 成为科学计算应用的主流语言。

NumPy 实现了多维同质数组 (homogeneous array) 和矩阵，这些数据结构不但能处理数字，
还能存放其他由用户定义的记录。通过 NumPy,用户能对这些数据结构里的元素进行高效的操作。

SciPy 是基于 NumPy 的另一个库，它提供了很多跟科学计算有关的算法，专为线性代数、数值积分和统计学而设计
SciPy 的高效和可靠性归功于其背后的 C 和 Fortran 代码，
而这些跟计算有关的部分都源自于 Netlib 库 (http://www.netlib.org)
SciPy 把基于 C 和 Fortran 的工业级数学计算功能用交互式且高度抽象的 Python 包装起来，让科学家如鱼得水

Pandas (http://pandas.pydata.org) 和 Blaze (http://blaze.pydata.org) 数据分析库就以它们为基础，
提供了高效的且能存储非数值类数据的数组类型，和读写常见数据文件格式(例如 csv、 xls、 SQL 转储和 HDF5)的功能
"""

import random
import array
import time
import numpy as np


# ----------------------------
if __name__ == "__main__":
    np_array = np.arange(12)
    print(type(np_array))
    print(np_array)
    print(np_array.shape)

    np_array.shape = 3, 4
    print(np_array)
    print(np_array[2])
    print(np_array[2, 1])
    print(np_array[:, 1])

    # 行列交换操作, 转置操作, 新的数组(原数组不变)
    print(np_array.transpose())
    print(np_array)

    # NumPy 可以对 numpy.ndarray 中的元素进行抽象的读取和保存和其他操作
    float_array = array.array('d', (random.random() for i in range(10**7)))
    np.savetxt('floats_10M_lines.txt', float_array)

    floats = np.loadtxt('floats_10M_lines.txt')
    print(floats[-3:])
    floats *= 0.5
    print(floats[-3:])

    time_start = time.perf_counter()
    floats /= 3
    time_end = time.perf_counter()
    print(f"the cost time is : {time_end - time_start}")

    np.save('floats_10M.npy', floats)
    #  numpy.load 方法利用了一种叫作内存映射的机制，在内存不足的情况下仍然可以对数组做切片
    floats_2 = np.load('floats_10M.npy', 'r+')
    floats_2 *= 6
    print(type(floats_2))
    print(floats_2[-3:])