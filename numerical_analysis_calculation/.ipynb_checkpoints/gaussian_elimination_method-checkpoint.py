#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: gaussian_elimination_method.py
# @copyright: https://gitee.com/weili_yzzcq/MachineLearning/numerical_analysis_calculation/
# @copyright: https://github.com/2694048168/MachineLearning/numerical_analysis_calculation/
# @function: 实现高斯消元法求解线性方程组，以及简单的测试

import numpy as np

def retroactive_resolution(coefficients: np.matrix, vector: np.array) -> np.array:
    """Gaussian elimination method for solving a system of linear equations.
    This function performs a retroactive linear system resolution for triangular matrix.
    
    高斯消去法的回代计算过程，即求解等价三角矩阵并得出对应位置未知数的解。
    创建一个新矩阵来储存线性方程组的解，而不是重写vector向量来覆盖原来的值。
    
    Args:
        coefficients: The coefficient matrix of the equivalent triangular matrix equations.
        vector: The value vector of the euqationequivalent triangular matrix equations.
    
    Returns:
        The vector of unknown parameters "X".

    Examples:
        2x1 + 2x2 - 1x3 = 5         
        0x1 - 2x2 - 1x3 = -7        
        0x1 + 0x2 + 5x3 = 15
    >>> gaussian_elimination([[2, 2, -1], [0, -2, -1], [0, 0, 5]], [[5], [-7], [15]])
    array([[2.],
           [2.],
           [3.]])
           
    Example:
        2x1 + 2x2 = -1
        0x1 - 2x2 = -1
    >>> gaussian_elimination([[2, 2], [0, -2]], [[-1], [-1]])
    array([[-1. ],
           [ 0.5]])
    """
    # 获取系数矩阵的行数和列数
    rows, columns = np.shape(coefficients)
    
    # 构建新矩阵用于存储方程组的解，可以使用 double 提高解的精度
    x = np.zeros((rows, 1), dtype=float)
    # 回代过程计算，是从最后一行开始计算的
    # 本程序的处理是从 k = n, n-1, n-2, ... , 2, 1
    for row in reversed(range(rows)):
        sum = 0
        for col in range(row + 1, columns):
            sum += coefficients[row, col] * x[col]
        
        # 求解每一行对应的未知数的解，储存到 x 列向量中【其实是一个矩阵，故此不能直接索引出来进行数值计算】
        # 虽然python是一种弱类型解释器，但是对于编写者而言，熟悉每一个变量的数据类型是必备的技能之一
        x[row, 0] = (vector[row] - sum) / coefficients[row, row]

    return x

def gaussian_elimination(coefficients: np.matrix, vector: np.array) -> np.array:
    """Gaussian elimination method for solving a system of linear equations.
    This function performs Gaussian elimination method.
    
    高斯消去法的消元计算过程，即将一个线性方程组的系数矩阵等价转换为一个上三角矩阵。
    
    Args:
        coefficients: The coefficient matrix of linear equations.
        vector: The value vector of the linear euqations.
    
    Returns:
        The vector of unknown parameters "X".
    
    Examples:
        1x1 - 4x2 - 2x3 = -2        
        5x1 + 2x2 - 2x3 = -3        
        1x1 - 1x2 + 0x3 = 4
    >>> gaussian_elimination([[1, -4, -2], [5, 2, -2], [1, -1, 0]], [[-2], [-3], [4]])
    array([[ 2.3 ],
           [-1.7 ],
           [ 5.55]])
           
    Examples:
        1x1 + 2x2 = 5
        5x1 + 2x2 = 5
    >>> gaussian_elimination([[1, 2], [5, 2]], [[5], [5]])
    array([[0. ],
           [2.5]])
    """
    # coefficients must to be a square matrix so we need to check first,
    # and then the linear equations can be solved.
    rows, columns = np.shape(coefficients)
    if rows != columns:
        print("The linear equations could not be solved for non-square matrix.")
        return []

    # augmented matrix 增广矩阵
    # 数据的合并是一种常见操作，对于高纬度数据而言，可以在指定轴上进行拼接操作
    # 对于二维数据而言，axis0 就是列轴；axis1 就是行轴
    augmented_mat = np.concatenate((coefficients, vector), axis=1)
    augmented_mat = augmented_mat.astype("float64")

    # scale the matrix leaving it triangular
    # 只需要进 n-1 次消元计算过程
    for row in range(rows - 1):
        # 每一次消元过程中每一行的第一个非零值元素
        pivot = augmented_mat[row, row]
        for col in range(row + 1, columns):
            # the multiplier（乘数） of each gaussian elimination
            factor = augmented_mat[col, row] / pivot
            augmented_mat[col, :] -= factor * augmented_mat[row, :]

    # 对变换后的增广矩阵进行切片索引
    # 分离出系数矩阵和值向量，这种操作经常出现在机器学习中对数据集的简单划分
    x = retroactive_resolution(augmented_mat[:, 0:columns], augmented_mat[:, columns : columns + 1])

    return x


if __name__ == "__main__":
    coefficient_matrix = np.matrix([[1, -4, -2], [5, 2, -2], [1, -1, 0]])
    value_vector = np.array([[-2], [-3], [4]])
    
    X = gaussian_elimination(coefficient_matrix, value_vector)
    
    X2 = gaussian_elimination([[1, 2], [5, 2]], [[5], [5]])
    
    X3 = gaussian_elimination([[0.6428, 0.3475, -0.8468], [0.3475, 1.8423, 0.4759], [-0.8468, 0.4759, 1.2147]], [[0.4127], [1.7321], [-0.8621]])

    print("The solution of this linear equations: \n", X)
    print("The solution of this linear equations: \n", X2)
    print("The solution of this linear equations: \n", X3)
    