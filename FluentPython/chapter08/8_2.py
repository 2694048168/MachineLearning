#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 8-2 元组的相对不可变性; 默认浅拷贝; 深拷贝
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-26

# 元组与多数 Python 集合（列表、字典、集，等等）一样，保存的是对象的引用
如果引用的元素是可变的，即便元组本身不可变，元素依然可变。
也就是说，元组的不可变性其实是指 tuple 数据结构的物理内容（即保存的引用）不可变，与引用的对象无关

# 复制对象时，相等性和标识之间的区别有更深入的影响。
副本与源对象相等，但是 ID 不同。
可是，如果对象中包含其他对象，那么应该复制内部对象吗？可以共享内部对象吗？

# 为任意对象做深复制和浅复制
浅复制没什么问题，但有时需要的是深复制(即副本不共享内部对象的引用)
copy 模块提供的 deepcopy 和 copy 函数能为任意对象做深复制和浅复制。
"""

import copy


# 校车乘客在途中上车和下车
class Bus():
    def __init__(self, passengers=None):
        if passengers is None:
            self.passengers = []
        else:
            self.passengers = list(passengers)

    def pick(self, name):
        self.passengers.append(name)
    
    def drop(self, name):
        self.passengers.remove(name)


# ----------------------------
if __name__ == "__main__":
    tuple_sequence_1 = (1, 2, [30, 40])
    tuple_sequence_2 = (1, 2, [30, 40])
    print(tuple_sequence_1 == tuple_sequence_2)
    print(id(tuple_sequence_1[-1]))

    tuple_sequence_1[-1].append(99)
    print(tuple_sequence_1)
    print(id(tuple_sequence_1[-1])) # 元组的元素的标识没有改变, 引用
    print(tuple_sequence_1 == tuple_sequence_2)

    # 默认浅拷贝 shallow copy
    print("-------- shallow copy --------")
    list_sequence_1 = [3, [55, 44], (7, 8, 9)]
    list_sequence_2 = list(list_sequence_1)
    print(list_sequence_2)
    print(list_sequence_2 == list_sequence_1)
    print(list_sequence_2 is list_sequence_1)

    # https://pythontutor.com/visualize.html#mode=edit
    l1 = [3, [66, 55, 44], (7, 8, 9)]
    l2 = list(l1)
    l1.append(100)
    l1[1].remove(55)
    print('l1:', l1)
    print('l2:', l2)
    l2[1] += [33, 22]
    l2[2] += (10, 11)
    print('l1:', l1)
    print('l2:', l2)

    # 为任意对象做深复制和浅复制
    print("-------- deep copy --------")
    bus_1 = Bus(['Alice', 'Bill', 'Claire', 'David'])
    bus_2 = copy.copy(bus_1)
    bus_3 = copy.deepcopy(bus_1)
    print(f"bus_1: {id(bus_1)}; bus_2: {id(bus_2)}; bus_3: {id(bus_3)}")

    bus_1.drop('Bill')
    print(bus_2.passengers)
    print(f"bus_1: {id(bus_1.passengers)}; bus_2: {id(bus_2.passengers)}; bus_3: {id(bus_3.passengers)}")
    print(bus_3.passengers)

    # 循环引用： b 引用 a，然后追加到 a 中； deepcopy 会想办法复制 a
    # https://docs.python.org/3/library/copy.html
    # 实现特殊方法 __copy__() 和 __deepcopy__()，控制 copy 和 deepcopy 的行为
    print("----------------------------")
    a = [10, 20]
    b = [a, 30]
    a.append(b)
    print(a)

    c = copy.deepcopy(a)
    print(c)