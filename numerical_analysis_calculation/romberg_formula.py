#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: romberg_formula.py
# @copyright: https://gitee.com/weili_yzzcq/MachineLearning/numerical_analysis_calculation/
# @copyright: https://github.com/2694048168/MachineLearning/numerical_analysis_calculation/
# @function: Romberg 公式求解数值积分，以及简单的测试

import math

# 被积函数 或者 某些函数值点
def integrand_function(x):
    # 避免出现除以 0 的情况
    if x == 0:
      return 1
    else:
      return math.sin(x) / x

# 复合梯形公式
def romberg_formula(integration_interval_a, integration_interval_b, integrand_function, sepsilon):
    """Compound trapezoidal formula.
    
     Romberg 公式求解数值积分，通过计算主对角线的元素之差作为误差估计，达到所需的精度即可完成数值积分，所求差值即所求的积分值。
    
    Args:
        integration_interval_a: The lower limit of integration interval.
        integration_interval_b: The upper limit of integration interval.
        integrand_function: The function to be integrated.
        sepsilon：The required calculation accuracy requirements.
    
    Returns:
        sum_R: The approximate value of romberg formula of actual integral value.
    
    Examples:
        math.sin(x) / x where [0, 1]
    >>> compound_trapezoidal_formula(0, 1, math.sin(x) / x, 1e-7)
     0.9456908635827014
    """
    # 计算 T_1
    T_1 = 1 / 2 * ( integrand_function(integration_interval_a) + integrand_function(integration_interval_b) )
    # 当前步长
    current_step_h = integration_interval_b - integration_interval_a
    
    # 计算 T_2 = T_1 + 新增加的计算点
    T_2 = 1 / 2 * T_1 + current_step_h / 2 * integrand_function((integration_interval_a + integration_interval_b) / 2)
    # 当前步长
    current_step_h = (integration_interval_b - integration_interval_a ) / 2
    
    # 计算 S_1
    S_1 = (4 * T_2 - T_1 ) / (4 - 1)
    
    # 计算误差精度
    if ((S_1 - T_1) < sepsilon):
        return S_1
    
    # 计算 T_4
    T_4 = 1 / 2 * T_2 + current_step_h / 2 * (integrand_function((integration_interval_a + integration_interval_b) / 4) + integrand_function((integration_interval_a + integration_interval_b)*3 / 4))
    # 当前步长
    current_step_h = (integration_interval_b - integration_interval_a ) / 4
    
    # 计算 S_2
    S_2 = (4 * T_4 - T_2 ) / (4 - 1)
    
    # 计算 C_1
    C_1 = (4**2 * S_2 - S_1 ) / (4**2 - 1)
    
    # 计算误差精度
    if ((C_1 - S_1) < sepsilon):
        return C_1
    
    # 计算 T_8
    T_8 = 1 / 2 * T_4 + current_step_h / 2 * (integrand_function((integration_interval_a + integration_interval_b) / 8) + 
                                      integrand_function((integration_interval_a + integration_interval_b)*3 / 8) + 
                                      integrand_function((integration_interval_a + integration_interval_b)*5 / 8) + 
                                      integrand_function((integration_interval_a + integration_interval_b)*7 / 8) )
    # 计算 S_4
    S_4 = (4 * T_8 - T_4 ) / (4 - 1)
    
    # 计算 C_2
    C_2 = (4**2 * S_4 - S_2 ) / (4**2 - 1)
    
    # 计算 R_1
    R_1 = (math.pow(4, 3) * C_2 - C_1 ) / (math.pow(4, 3) - 1)
    
    # 计算误差精度
    if ((R_1 - C_1) < sepsilon):
        return R_1
    
    # 计算失败    
    return 0


# test
if __name__ == "__main__":
    # 积分区间下限
    integration_interval_a = 0
    # 积分区间上限
    integration_interval_b = 1
    # 积分区间等间隔划分
    sepsilon = 1e-7
    
    sum_R = romberg_formula(integration_interval_a, integration_interval_b, integrand_function, sepsilon)
    print("The result of the romberg formula for the integrand function : \n", sum_R)