#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 2-4 元组: 不可变列表和没有字段名的记录
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-17
"""

import os
from collections import namedtuple


# ----------------------------
if __name__ == "__main__":
    # 1. 元组和记录(位置信息和数量信息)
    lax_coordinates = (33.94255, -118.408056)
    city, year, pop, chg, area = ("Tokyo", 2003, 32450, 0.66, 8014)
    traveler_ids = [('USA', "31195855"), ("BRA", "CE342567"), ("ESP", "XDA205856")]
    for passport in sorted(traveler_ids):
        # % 格式运算符能被匹配到对应的元组元素上
        print('%s|%s' % passport)

    # for 循环可以分别提取元组里的元素，也叫作拆包 unpacking
    # 因为元组中第二个元素没有什么用，所以它赋值给 “_” 占位符
    for country, _ in traveler_ids:
        print(country)

    # 拆包 unpacking 让元组 tuple 可以完美被当作记录来使用
    # 2. 元组拆包
    # style-1 元组拆包形式就是平行赋值，把一个可迭代对象里的元素，一并赋值到由对应的变量组成的元组
    latitude, longitude = lax_coordinates
    print(latitude)
    print(longitude)

    # style-2 优雅的写法当属不使用中间变量交换两个变量
    a, b = 24, 42
    print(a, b)
    a, b = b, a
    print(a, b)

    # style-3 用 * 运算符把一个可迭代对象拆开作为函数的参数X
    print(divmod(20, 8))
    param_tuple = (20, 8)
    print(divmod(*param_tuple))
    quotient, remainder = divmod(*param_tuple)
    print(quotient, remainder)

    # 元组拆包的用法则是让一个函数可以用元组的形式返回多个值，
    # 然后调用函数的代码就能轻松地接受这些返回值。
    # 比如 os.path.split() 函数就会返回以路径和最后一个文件名组成的元组 (path, last_part):
    _, filename = os.path.split("./../chapter01/1_1.py")
    print(filename)
    # 如果做的是国际化软件，那么 _ 可能就不是一个理想的占位符，因为它也是 gettext.gettext 函数的常用别名， 
    # gettext 模块的文档（https://docs.python.org/3/library/gettext.html）里提到了这一点。
    # 在其他情况下， _ 会是一个很好的占位符。

    # 在元组拆包中使用 * 也可以帮助把注意力集中在元组的部分元素上。
    # 用 * 来处理剩下的元素
    # 在 Python 中，函数用 *args 来获取不确定数量的参数算是一种经典写法了。
    # 于是 Python 3 里，这个概念被扩展到了平行赋值中
    a, b, *rest = range(5)
    print(a, b, rest)
    a, b, *rest = range(3)
    print(a, b, rest)
    a, b, *rest = range(2)
    print(a, b, rest)

    # 在平行赋值中， * 前缀只能用在一个变量名前面，
    # 但是这个变量可以出现在赋值表达式的任意位置：
    a, *body, c, d = range(5)
    print(a, body, c, d)
    *head, b, c, d = range(5)
    print(head, b, c, d)

    # style-4 嵌套元组拆包
    metro_areas = [
        ('Tokyo','JP',36.933,(35.689722,139.691667)),
        ('Delhi NCR', 'IN', 21.935, (28.613889, 77.208889)),
        ('Mexico City', 'MX', 20.142, (19.433333, -99.133333)),
        ('New York-Newark', 'US', 20.104, (40.808611, -74.020386)),
        ('Sao Paulo', 'BR', 19.649, (-23.547778, -46.635833)),
        ]
    print(f"{'':15} | {'lat':^9} | {'long':^9}")
    # format_string = "{:15} | {:9.4f} | {:9.4f}"
    for name, cc, pop, (latitude, longitude) in metro_areas:
        if longitude <= 0:
            # print(format_string.format(name, latitude, longitude))
            print(f"{name:15} | {latitude:9.4f} | {longitude:9.4f}")