#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: newton_method.py
# @copyright: https://gitee.com/weili_yzzcq/MachineLearning/numerical_analysis_calculation/
# @copyright: https://github.com/2694048168/MachineLearning/numerical_analysis_calculation/
# @function: 实现牛顿迭代法求解方程的根，以及简单的测试

import math

def Newton_Method(x0, epsilon, number_iteration, function, derivative_function):
    """Implementation of Newton Method for solving equations
    
    使用 Newton Method 可能会出现收敛不了的情况，即就是发散了。
    所以需要设置一个迭代次数的上限来结束迭代过程。
    
    Args:
        x0: The initial value to equation for Newton Method.
        epsilon: The error between the previous iteration and the next iteration.
        number_iteration: The maximum number of iterations.
        function: The equation function for solving.
        derivative_function: The derivative function to the equation function for solving.
        
    Returns:
        The root to this equation function "function" with Newton Method.
        
    Examples:
    >>> Newton_Method(1.0, 1e-6, 20, math.pow(x, 3) - x - 1, 2*x*x - 1)
    1.324717957244746
    """
    f0 = function(x0)
    df0 = derivative_function(x0)

    iteration = 0
    while (iteration < number_iteration):
        # Newton 迭代式
        x1 = x0 - f0 / df0
        
        f1 = function(x1)
        df1 = derivative_function(x1)
        
        # 绝对误差限 |x1 - x0|
        # 相对误差限 | (x1-x0)/x1 |
        if abs(x1) < 1:
            delta = abs(x1 - x0)
        else:
            delta = abs(x1 - x0) / abs(x1)
        if delta < epsilon:
            break
        else:
            x0 = x1
            f0 = f1
            df0 = df1

            ++iteration

    return x1
    
def function(x):
    return math.pow(x, 3) - x - 1
  
def derivative_function(x): 
    return 2 * x * x - 1 

if __name__ == '__main__':
    x0 = 1.0
    epsilon = 1.e-10
    number_iteration = 20
    
    x = Newton_Method(x0, epsilon, number_iteration, function, derivative_function)
    
    print("函数方程的近视解为：", x)
