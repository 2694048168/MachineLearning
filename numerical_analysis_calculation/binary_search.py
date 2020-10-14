#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: binary_search.py
# @copyright: https://gitee.com/weili_yzzcq/MachineLearning/numerical_analysis_calculation/
# @copyright: https://github.com/2694048168/MachineLearning/numerical_analysis_calculation/
# @function: 实现二分法对方程的根的搜索算法，以及简单的测试

import math

def Binary_Search(a, b, epsilon, function):
    """Get a root of function with using Binary Search.
    
    二分法进行对根的搜索，该方法一定会收敛，故此不需要设置迭代次数。
    如果需要寻找一个合适的初值，则将误差限设置大一点即可，
    返回值即可作为其他方法（Newton Method，牛顿迭代法具有局部收敛性）的迭代初始值。
    
    Args:
        a: The start of the real root interval.
        b: The end of the real root interval.
        epsilon: The error between the real solution and the myopic solution.
        function: The equation function for root.
    
    Returns:
        The root to this equation function "function".
    
    Examples:
    >>> Binary_Search(1.0, 1.5, 1e-6, math.pow(x, 3) - x - 1)
    1.25
    """
    binary_root = 0
    a_value = function(a)
    b_value = function(b)
    
    if a_value * b_value > 0:
        # 不满足数学中的 “ 零点定理 ” ，抛出异常
        raise ValueError("The function must have different sign at a and b.")
    
    while True:
        binary_root = (a + b) / 2
        binary_value = function(binary_root)
        # 迭代满足误差需求或者求得解析解，则直接退出迭代
        if ((a - b) / 2 < epsilon) or (binary_value == 0):
            break
        # 不断迭代更新有根区间
        if a_value * binary_value < 0:
            b = binary_root
        elif b_value * binary_value < 0:
            a = binary_root
        
    return binary_root


def function(x):
    return math.pow(x, 3) - x - 1


if __name__ == '__main__':
    a = 1.0
    b = 1.5
    epsilon = 0.005
    
    x = Binary_Search(a, b, epsilon, function)
    
    print("函数方程的近视解为：", x)
