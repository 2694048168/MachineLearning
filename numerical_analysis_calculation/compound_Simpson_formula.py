#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: compound_Simpson_formula.py
# @copyright: https://gitee.com/weili_yzzcq/MachineLearning/numerical_analysis_calculation/
# @copyright: https://github.com/2694048168/MachineLearning/numerical_analysis_calculation/
# @function: 实现复合 Simpson 公式求积分方法，以及简单的测试

import math

# 被积函数 或者 某些函数值点
def integrand_function(x):
    # 避免出现除以 0 的情况
    if x == 0:
      return 1
    else:
      return math.sin(x) / x

# 复合 Simpson 公式
def compound_Simpson_formula(integration_interval_a, integration_interval_b, iter_num):
    """Compound trapezoidal formula.
    
    使用 Simpson 公式进行每一个子区间的求积分，然后累计求和来近似所求的积分。
    
    Args:
        integration_interval_a: The lower limit of integration interval.
        integration_interval_b: The upper limit of integration interval.
        iter_num: Divide the integration interval at equal intervals.
    
    Returns:
        sum_S: The approximate value of compound Simpson formula of actual integral value.
    
    Examples:
        math.sin(x) / x where [0, 1]
    >>> compound_Simpson_formula(0, 1, 4)
     0.9460833108884719
    """
    # 步长间隔
    step_h = (integration_interval_a + integration_interval_b) / iter_num
    sum_S = 0.0
    for k in range(iter_num):
        # Simpson 公式计算
        sum_S += (integrand_function(integration_interval_a + k * step_h) + integrand_function(integration_interval_a + (k+1) * step_h) +
                  4 * integrand_function(integration_interval_a + k * step_h + step_h / 2)) * step_h / 6
        
    return sum_S

# 误差分析，积分余项

# test
if __name__ == "__main__":
    # 积分区间下限
    integration_interval_a = 0
    # 积分区间上限
    integration_interval_b = 1
    # 积分区间等间隔划分
    iter_num = 4
    
    sum_S = compound_Simpson_formula(integration_interval_a, integration_interval_b, iter_num)
    print("The result of the compound Simpson formula for the integrand function : \n", sum_S)