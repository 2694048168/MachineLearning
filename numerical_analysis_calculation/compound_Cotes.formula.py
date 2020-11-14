#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: compound_Cotes_formula.py
# @copyright: https://gitee.com/weili_yzzcq/MachineLearning/numerical_analysis_calculation/
# @copyright: https://github.com/2694048168/MachineLearning/numerical_analysis_calculation/
# @function: 实现复合 Cotes 公式求积分方法，以及简单的测试

import math

# 被积函数 或者 某些函数值点
def integrand_function(x):
    # 避免出现除以 0 的情况
    if x == 0:
      return 1
    else:
      return math.sin(x) / x

# 复合 Cotes 公式
def compound_Cotes_formula(integration_interval_a, integration_interval_b, iter_num):
    """Compound trapezoidal formula.
    
    使用 Cotes 公式进行每一个子区间的求积分，然后累计求和来近似所求的积分。
    
    Args:
        integration_interval_a: The lower limit of integration interval.
        integration_interval_b: The upper limit of integration interval.
        iter_num: Divide the integration interval at equal intervals.
    
    Returns:
        sum_C: The approximate value of compound Cotes formula of actual integral value.
    
    Examples:
        math.sin(x) / x where [0, 1]
    >>> compound_Cotes_formula(0, 1, 2)
     1.0238608471286947
    """
    # 步长间隔
    step_h = (integration_interval_a + integration_interval_b) / iter_num
    sum_C = 0.0
    sum_inter = 0.0
    for k in range(iter_num):
        # Cotes 公式中需要累加计算的子式子
        sum_inter += (32 * integrand_function(integration_interval_a + k * step_h + step_h / 4) + 
                12 * integrand_function(integration_interval_a + k * step_h + step_h / 2) + 
                32 * integrand_function(integration_interval_a + k * step_h + step_h * 3 / 4) +
                14 * integrand_function(integration_interval_a + k * step_h))
    
    # Cotes 公式计算                
    sum_C = (7 * integrand_function(integration_interval_a) + 7 * integrand_function(integration_interval_b) + sum_inter) * step_h / 90
    
    return sum_C

# 误差分析，积分余项

# test
if __name__ == "__main__":
    # 积分区间下限
    integration_interval_a = 0
    # 积分区间上限
    integration_interval_b = 1
    # 积分区间等间隔划分
    iter_num = 2
    
    sum_C = compound_Cotes_formula(integration_interval_a, integration_interval_b, iter_num)
    print("The result of the compound Simpson formula for the integrand function : \n", sum_C)