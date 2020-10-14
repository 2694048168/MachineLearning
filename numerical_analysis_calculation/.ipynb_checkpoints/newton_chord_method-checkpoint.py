#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: newton_chord_method.py
# @copyright: https://gitee.com/weili_yzzcq/MachineLearning/numerical_analysis_calculation/
# @copyright: https://github.com/2694048168/MachineLearning/numerical_analysis_calculation/
# @function: 实现牛顿迭代法和弦截法代替一阶导数结合求解方程的根，以及简单的测试

import math

def Newton_Chord_Method(x0, x1, epsilon, number_iteration, function):
    """Implementation of Newton Chord Method for solving equations
    
    使用 Newton Method 可能会出现收敛不了的情况，即可能出现发散了。
    所以需要设置一个迭代次数的上限来结束迭代过程。
    
    Args:
        x0: The first initial value to equation for Newton Chord Method.
        x2: The secondary initial value to equation for Newton Chord Method.
        epsilon: The error between the previous iteration and the next iteration.
        number_iteration: The maximum number of iterations.
        function: The equation function for solving.
        
    Returns:
        The root to this equation function "function" with Newton Chord Method.
        
    Examples:
    >>> Newton_Chord_Method(1.0, 1.05, 1e-6, 20, math.pow(x, 3) - x - 1)
    1.1572463768215488
    """
    f0 = function(x0)
    f1 = function(x1)
    
    iteration = 0
    while (iteration < number_iteration):
        x2 = x1 - f1 / (f1 + f0) * (x1 - x0)
        
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
            x1 = x2
            f0 = function(x1)
            f1 = function(x2)
            
            ++iteration
    
    return x2


def function(x):
    return math.pow(x, 3) - x - 1
  

if __name__ == '__main__':
    x0 = 1.0
    x1 = 1.2
    epsilon = 1.e-10
    number_iteration = 20
    
    x = Newton_Chord_Method(x0, x1, epsilon, number_iteration, function)
    
    print("函数方程的近视解为：", x)
