#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 5-6 支持函数时编程的 Python 包
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-23

Python 也支持函数式编程风格
# 1. operator 模块
# 2. functools 模块

在函数式编程中，经常需要把算术运算符当作函数使用
"""

import operator
import functools
import collections

# 1. 使用 reduce 函数和一个匿名函数计算阶乘
def fact_1(n):
    return functools.reduce(lambda a, b: a * b, range(1, n + 1))

# 2. 使用 reduce 和 operator.mul 函数计算阶乘
def fact_2(n):
    return functools.reduce(operator.mul, range(1, n + 1))


# ----------------------------
if __name__ == "__main__":
    print("-------------------------------")
    print(fact_1(5))
    print(fact_2(5))

    # 演示使用 itemgetter 排序一个元组列表
    print("-------------------------------")
    metro_data = [
        ('Tokyo', 'JP', 36.933, (35.689722, 139.691667)),
        ('Delhi NCR', 'IN', 21.935, (28.613889, 77.208889)),
        ('Mexico City', 'MX', 20.142, (19.433333, -99.133333)),
        ('New York-Newark', 'US', 20.104, (40.808611, -74.020386)),
        ('Sao Paulo', 'BR', 19.649, (-23.547778, -46.635833)),
        ]
    for city in sorted(metro_data, key=operator.itemgetter(1)):
        print(city)

    print("-------------------------------")
    cc_name = operator.itemgetter(1, 0)
    for city in metro_data:
        print(cc_name(city))

    print("-------------------------------")
    # 定义一个 namedtuple，名为 metro_data 演示使用 attrgetter 处理它
    LatLong = collections.namedtuple('LatLong', 'lat long')
    Metropolis = collections.namedtuple('Metropolis', 'name cc pop coord')
    metro_areas = [Metropolis(name, cc, pop, LatLong(lat, long)) 
                    for name, cc, pop, (lat, long) in metro_data]

    print(metro_areas[0])
    print(metro_areas[0].coord.lat)

    name_lat = operator.attrgetter('name', 'coord.lat')
    for city in sorted(metro_areas, key=operator.attrgetter('coord.lat')):
        print(name_lat(city))

    print("-------------------------------")
    print([name for name in dir(operator)])
    print("-------------------------------")
    print([name for name in dir(operator) if not name.startswith('_')])

    # 在 operator 模块余下的函数中 methodcaller
    # 它的作用与 attrgetter 和 itemgetter 类似，它会自行创建函数
    # methodcaller 创建的函数会在对象上调用参数指定的方法
    print("-------------------------------")
    s = 'The time has come'
    up_case = operator.methodcaller('upper')
    print(s)
    print(up_case(s))

    hiphenate = operator.methodcaller('replace', ' ', '-')
    print(hiphenate(s))

    # 使用 functools.partial 冻结参数
    # functools.partial 这个高阶函数用于部分应用一个函数。
    # 部分应用是指，基于一个函数创建一个新的可调用对象，把原函数的某些参数固定。
    # 使用这个函数可以把接受一个或多个参数的函数改编成需要回调的 API，这样参数更少。
    print("-------------------------------")
    # 使用 partial 把一个两参数函数改编成需要单参数的可调用对象
    triple = functools.partial(operator.mul, 3)
    print(triple(7))
    print(list(map(triple, range(1, 10))))
    # functools 模块中的 lru_cache 函数令人印象深刻，它会做备忘 (memoization)
    # 这是一种自动优化措施，它会存储耗时的函数调用结果，避免重新计算