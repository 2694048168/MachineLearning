#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 5-1 将函数视为对象
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-22
"""

def factorial(num):
    """return n!
    Args:
        num (int): 计算整数 num 的阶乘
    Returns:
        int: 计算整数 num 的阶乘
    """
    return 1 if num < 2 else num * (factorial(num - 1))

# ----------------------------
if __name__ == "__main__":
    print(factorial(42))
    print(factorial(1))
    print(factorial.__doc__)
    print(type(factorial))

    # 一等函数特性 
    fact = factorial
    print(fact)
    print(fact(5))
    print(map(factorial, range(11)))
    print(list(map(factorial, range(11))))
    # 函数式风格编程。函数式编程的特点之一是使用高阶函数