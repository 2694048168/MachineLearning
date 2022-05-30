#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 7-3 闭包
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-24

闭包指延伸了作用域的函数，其中包含函数定义体中引用、但是不在定义体中定义的非全局变量。
函数是不是匿名的没有关系，关键是它能访问定义体之外定义的非全局变量。

闭包是一种函数，它会保留定义函数时存在的自由变量的绑定，
这样调用函数时，虽然定义作用域不可用了，但是仍能使用那些绑定。
注意，只有嵌套在其他函数中的函数才可能需要处理不在全局作用域中的外部变量
"""

# 计算移动平均值的类
class Averager():
    def __init__(self):
        self.series = []

    def __call__(self, new_value):
        self.series.append(new_value)
        total = sum(self.series)
        return total/len(self.series)

# 计算移动平均值的高阶函数
def make_averager():
    # 注意， series 是 make_averager 函数的局部变量
    series = []

    def averager(new_value):
        # 在 averager 函数中， series 是自由变量 free variable
        series.append(new_value)
        total = sum(series)
        return total/len(series)
    
    return averager


# ----------------------------
if __name__ == "__main__":
    print("-------- Averager Class ---------")
    avg = Averager()
    print(avg(10))
    print(avg(11))
    print(avg(12))

    print("--------  make averager function ---------")
    avg_func = make_averager()
    print(avg_func(10))
    print(avg_func(11))
    print(avg_func(12))

    print("-------- Python __code__  ---------")
    print(f"function local variables : {avg_func.__code__.co_varnames}")
    print(f"function free variables : {avg_func.__code__.co_freevars}")
    # series 的绑定在返回的 avg 函数的 __closure__ 属性中
    #  avg.__closure__ 中的各个元素对应于 avg.__code__.co_freevars 中的一个名称
    # 这些元素是 cell 对象，有个 cell_contents 属性，保存着真正的值
    print(avg_func.__closure__)
    print(avg_func.__closure__[0].cell_contents)