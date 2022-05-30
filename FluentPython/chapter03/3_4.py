#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 3-4 __missing__ 处理缺失键的情况
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-21

有时候为了方便起见，就算某个键在映射里不存在，也希望在通过这个键读取值的时候能得到一个默认值。
1. 是通过 defaultdict 这个类型而不是普通的 dict
    collections.defaultdict 创建 defaultdict 对象的时候，
    就需要给它配置一个为找不到的键创造默认值的方法。
    具体而言，在实例化一个 defaultdict 的时候，需要给构造方法提供一个可调用对象，
    这个可调用对象会在 __getitem__ 碰到找不到的键的时候被调用，
    让 __getitem__ 返回某种默认值。
    而这个用来生成默认值的可调用对象存放在名为 default_factory 的实例属性里。

2. 是给自己定义一个 dict 的子类，然后在子类中实现 __missing__ 方法
    所有的映射类型在处理找不到的键的时候，都会牵扯到 __missing__ 方法。
    如果有一个类继承了 dict,然后这个继承类提供了 __missing__ 方法，
    那么在 __getitem__ 碰到找不到的键的时候， Python 就会自动调用它,而不是抛出一个KeyError 异常。
    __missing__ 方法只会被 __getitem__ 调用（比如在表达式 d[k] 中）
    提供__missing__ 方法对 get 或者 __contains__ (in 运算符会用到这个方法)
    这些方法的使用没有影响。defaultdict 中的 default_factory 只对 __getitem__ 有作用的原因。
"""

# ---------------------
import collections
# https://docs.python.org/3/library/collections.html
# 标准库 collections 模块中，有着许多不同的映射类型(字典的变种)
# 这个类其实就是把标准 dict 用纯 Python 又实现了一遍
# 1. collections.OrderedDict
# 2. collections.ChainMap
# 2. collections.Counter
# ---------------------


# ---- BEGIN STRKEYDICT0 ----
class StrKeyDict0(dict):  # <1>
    def __missing__(self, key):
        if isinstance(key, str):  # <2>
            raise KeyError(key)
        return self[str(key)]  # <3>

    def get(self, key, default=None):
        try:
            return self[key]  # <4>
        except KeyError:
            return default  # <5>

    def __contains__(self, key):
        return key in self.keys() or str(key) in self.keys()  # <6>
# ---- END STRKEYDICT0 ----

# ---- BEGIN STRKEYDICT0 ----
class StrKeyDict(collections.UserDict):
    def __missing__(self, key):
        if isinstance(key, str):
            return KeyError(key)
        return self[str(key)]

    def __contains__(self, key):
        return str(key) in self.data

    def __setitem__(self, key, item):
        self.data[str(key)] = item
# ---- END STRKEYDICT0 ----

# ----------------------------
if __name__ == "__main__":
    # 当有非字符串的键被查找的时候， 
    # StrKeyDict0 是如何在该键不存在的情况下，把它转换为字符串的
    dict_sequence = StrKeyDict([('2', 'two'), ('4', 'four')])
    # dict_sequence = StrKeyDict0([('2', 'two'), ('4', 'four')])

    # 1. __missing__ method
    print(dict_sequence['2'])
    print(dict_sequence['4'])
    # print(dict_sequence['1'])

    # 2. get method
    print(dict_sequence.get('2'))
    print(dict_sequence.get('4'))
    print(dict_sequence.get('1', 'N/A'))

    # 3. __contains__ method
    print(2 in dict_sequence)
    print(4 in dict_sequence)
    print(1 in dict_sequence)

    # https://docs.python.org/3/library/collections.html#collections.Counter
    counter = collections.Counter("abracadabra")
    print(counter)
    counter.update("aaaaazzz")
    print(counter)
    print(counter.most_common(2))