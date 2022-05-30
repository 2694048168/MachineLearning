#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 2-5 具名元组和作为不可变列表的元组
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-17
"""

from collections import namedtuple


# ----------------------------
if __name__ == "__main__":
    # 3. 具名元组
    # 元组已经设计得很好用了，但作为记录来用的话，还是少了一个功能：时常会需要给记录中的字段命名。 
    # namedtuple 函数的出现帮我们解决了这个问题。
    # collections.namedtuple 是一个工厂函数，它可以用来构建一个带字段名的元组和一个有名字的类——这个带名字的类对调试程序有很大帮助。
    # 用 namedtuple 构建的类的实例所消耗的内存跟元组是一样的，因为字段名都被存在对应的类里面。
    # 这个实例跟普通的对象实例比起来也要小一些，因为 Python 不会用 __dict__ 来存放这些实例的属性。
    City = namedtuple("City", "name country population coordinates")
    tokyo =City('Tokyo', "JP", 36.933, (35.689722, 139.691667))
    print(tokyo)

    # 通过字段名或者位置来获取一个字段的信息
    print(tokyo.population)
    print(tokyo.coordinates)
    print(tokyo[1])
    print(tokyo[0])

    # 除了从普通元组那里继承来的属性之外，具名元组还有一些自己专有的属性
    print(City._fields, "\n") # 类属性

    LatLong = namedtuple("LatLong", "lat long")
    delhi_data = ('Delhi NCR', 'IN', 21.935, LatLong(28.613889, 77.208889))
    delhi = City._make(delhi_data)
    print(delhi)
    print(City(*delhi_data), "\n")

    # _asdict() 把具名元组以 collections.OrderedDict 的形式返回，
    # 可以利用它来把元组里的信息友好地呈现出来。
    print(type(delhi._asdict()))
    print(delhi._asdict())
    for key, value in delhi._asdict().items():
        print(key + ":", value)
    print()

    # 元组作为不可变的列表
    # 如果要把元组当作列表来用的话，最好先了解一下它们的相似度如何。
    # 除了跟增减元素相关的方法之外，元组支持列表的其他所有方法。
    # 还有一个例外，元组没有 __reversed__ 方法，但是这个方法只是个优化而已， 
    # reversed(my_tuple) 这个用法在没有 __reversed__ 的情况下也是合法的。