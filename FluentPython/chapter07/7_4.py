#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 7-4 nonlocal
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-24

了解 Python 闭包，下面可以使用嵌套函数正式实现装饰器
"""

# 计算移动平均值的高阶函数，不保存所有历史值，但有缺陷
def make_averager():
    count = 0
    total = 0

    def averager(new_value):
        count += 1
        total += new_value
        return total / count
    return averager

# 计算移动平均值的高阶函数，不保存所有历史值，但有缺陷
def make_averager_nonlocal():
    count = 0
    total = 0

    def averager(new_value):
        # ----------------------
        # 使用 nonlocal 修正
        nonlocal count, total
        # ----------------------
        count += 1
        total += new_value
        return total / count
    return averager


# ----------------------------
if __name__ == "__main__":
    avg = make_averager()
    # print(avg(10))
    # --------------------------------------------------------------------------
    # 问题是，当 count 是数字或任何不可变类型时， 
    # count += 1 语句的作用其实与 count = count + 1 一样。
    # 因此在 averager 的定义体中为 count 赋值了，这会把 count 变成局部变量。 
    # total 变量也受这个问题影响。
    # 示例 7-3 没遇到这个问题，因为没有给 series 赋值，只是调用 series.append，
    # 并把它传给 sum 和 len。也就是说，利用了列表是可变的对象这一事实。
    # 但是对数字、字符串、元组等不可变类型来说，只能读取，不能更新。
    # 如果尝试重新绑定，例如 count = count + 1，其实会隐式创建局部变量 count。
    # 这样， count 就不是自由变量了，因此不会保存在闭包中。
    # 为了解决这个问题， Python 3 引入了 nonlocal 声明。
    # 它的作用是把变量标记为自由变量，即使在函数中为变量赋予新值了，也会变成自由变量。
    # 如果为 nonlocal 声明的变量赋予新值，闭包中保存的绑定会更新。
    # --------------------------------------------------------------------------
    avg_func = make_averager_nonlocal()
    print(avg_func(10))
    print(avg_func(11))
    print(avg_func(12))
    print(avg_func(24))
    print(avg_func(42))