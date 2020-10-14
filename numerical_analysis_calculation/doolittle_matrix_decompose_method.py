#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: doolittle_matrix_decompose_method.py
# @copyright: https://gitee.com/weili_yzzcq/MachineLearning/numerical_analysis_calculation/
# @copyright: https://github.com/2694048168/MachineLearning/numerical_analysis_calculation/
# @function: 实现Doolittle分解矩阵以及求解线性方程组，以及简单的测试

import numpy as np

def LU_matrix_decompose(matrix):
    """Lower-Upper (LU) Decomposition.
    
    将一个方阵进行 LU 分解，使用 Doolittle 分解法，
    其中，L 为单位下三角矩阵，U 为上三角矩阵。
    
    Args:
        matrix: The square matrix.
    
    Returns:
        The unit lower triangular matrix "L",
        and the upper triangular matrix "U".
    
    Examples:
        1x1 + 2x2 + 3x3 = 14       
        2x1 + 5x2 + 2x3 = 18        
        3x1 + 1x2 + 5x3 = 20
    >>> LU_matrix_decompose([[1, 2, 3], [2, 5, 2], [3, 1, 5]])
    array([[1., 0., 0.],
           [2., 1., 0.],
           [3., -5., 1.]])
    array([[1., 2., 3.],
           [0., 1, -4.],
           [0., 0, -24.]])
    """
    # Matrix has to be a square array so we need to check first.
    rows, columns = np.shape(matrix)
    if rows != columns:
        print("Error: must be a square matrix.")
        return []
    
    # L 为单位下三角矩阵， U 为上三角矩阵
    L = np.zeros((rows, columns))
    U = np.zeros((rows, columns))
    # computing the L matrix and the U matrix.
    for j in range(rows):
        # L 为单位下三角矩阵
        L[j][j] = 1.0
        # Computing the upper triangular matrix "U"
        # 上三角矩阵的行列索引关系：j(rows) >= i(columns)
        for i in range(j+1):
            sum_U = 0
            sum_U = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = matrix[i][j] - sum_U
        # Computing the unit lower triangular matrix "L"
        # 下三角矩阵的行列索引关系：j(rows) <= i(columns)
        for i in range(j, rows):
            sum_L = 0
            sum_L = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (matrix[i][j] - sum_L) / U[j][j]

    return L, U

def Doolittle_method(coefficient_matrix, vector):
    """Doolittle method for solving a system of linear equations.
    
    使用 Doolittle 方法求解一个线性方程组，首先将一个方阵进行 LU 分解，使用 Doolittle 分解法，
    其中，L 为单位下三角矩阵，U 为上三角矩阵，然后进行回代计算线性方程组的解。
    AX = b <==> LUX = b <==> 
    LY = b and UX = Y
    
    Args:
        coefficient_matrix: The coefficient matrix of linear equations.
        vector: The value vector of the linear euqations.
    
    Returns:
        X: The vector of unknown parameters for the linear equations "X".
        Y: The intermediate variable vector "Y".
    
    Examples:
        1x1 + 2x2 + 3x3 = 14       
        2x1 + 5x2 + 2x3 = 18        
        3x1 + 1x2 + 5x3 = 20
    >>> Doolittle_method([[1, 2, 3], [2, 5, 2], [3, 1, 5]], [[14], [18], [20]])
    array([[1.]
           [2.]
           [3.]])
    array([[ 14.]
           [-10.]
           [-72.]])
    """
    # Matrix has to be a square array so we need to check first.
    rows, columns = np.shape(matrix)
    if rows != columns:
        print("Error: must be a square matrix.")
        return []
    
    # Doolittle decomposition
    L, U = LU_matrix_decompose(coefficient_matrix)
    Y = np.zeros((rows, 1))
    X = np.zeros((rows, 1))
    
    # Computing the Y
    for i in range(rows):
        sum_Y = sum(L[i][k] * Y[k] for k in range(i))
        Y[i] = vector[i] - sum_Y
        
    # Computing the X
    for j in reversed(range(rows)):
        sum_X = sum(U[j][k] * X[k] for k in range(j+1, rows))
        X[j] = (Y[j] - sum_X) / U[j][j]
            
    return X, Y
    

if __name__ == "__main__":
    matrix = np.array([[1, 2, 3], [2, 5, 2], [3, 1, 5]])
    vector = np.array([[14], [18], [20]])
    
    L, U = LU_matrix_decompose(matrix)
    print("The L matrix:")
    print(L)
    print("The U matrix:")
    print(U)
    
    X, Y = Doolittle_method(matrix, vector)
    print("The X matrix:")
    print(X)
    print("The Y matrix:")
    print(Y)
