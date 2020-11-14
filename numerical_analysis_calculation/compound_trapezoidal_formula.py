#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: compound_trapezoidal_formula.py
# @copyright: https://gitee.com/weili_yzzcq/MachineLearning/numerical_analysis_calculation/
# @copyright: https://github.com/2694048168/MachineLearning/numerical_analysis_calculation/
# @function: 实现复合梯形求积分方法，以及简单的测试

import math

# 被积函数 或者 某些函数值点
def integrand_function(x):
    # 避免出现除以 0 的情况
    if x == 0:
      return 1
    else:
      return math.sin(x) / x

# 复合梯形公式
def compound_trapezoidal_formula(integration_interval_a, integration_interval_b, iter_num):
    """Compound trapezoidal formula.
    
    使用梯形公式进行每一个子区间的求积分，然后累计求和来近似所求的积分。
    
    Args:
        integration_interval_a: The lower limit of integration interval.
        integration_interval_b: The upper limit of integration interval.
        iter_num: Divide the integration interval at equal intervals.
    
    Returns:
        sum_T: The approximate value of compound trapezoidal formula of actual integral value.
    
    Examples:
        math.sin(x) / x where [0, 1]
    >>> compound_trapezoidal_formula(0, 1, 8)
     0.9456908635827014
    """
    # 步长间隔
    step_h = (integration_interval_a + integration_interval_b) / iter_num
    sum_T = 0.0
    for k in range(iter_num):
        # 梯形公式计算
        sum_T += (integrand_function(integration_interval_a + k * step_h) + integrand_function(integration_interval_a + (k+1) * step_h)) * step_h / 2
        
    return sum_T

# 误差分析，积分余项

# test
if __name__ == "__main__":
    # 积分区间下限
    integration_interval_a = 0
    # 积分区间上限
    integration_interval_b = 1
    # 积分区间等间隔划分
    iter_num = 8
    
    sum_T = compound_trapezoidal_formula(integration_interval_a, integration_interval_b, iter_num)
    print("The result of the compound trapezoidal formula for the integrand function : \n", sum_T)