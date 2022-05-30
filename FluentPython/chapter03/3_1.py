#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 3-1 泛映射类型, 字典构造函数, 字典推导式 dictcomp
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-21
"""

from collections import abc

# ----------------------------
if __name__ == "__main__":
    dict_sequence = {}
    print(isinstance(dict_sequence, abc.Mapping))
    print(type(dict_sequence))

    # 标准库里的所有映射类型都是利用 dict 来实现的，因此它们有个共同的限制，
    # 即只有可散列的数据类型才能用作这些映射里的键(只有键有这个要求，值并不需要是可散列的数据类型)
    # https://docs.python.org/3/glossary.html#term-hashable
    tuple_tuple_sequence = (1, 2, (30, 40))
    print(hash(tuple_tuple_sequence))

    # tuple_list_sequence = (1, 2, [30, 40])
    # print(hash(tuple_list_sequence))

    tuple_list_sequence = (1, 2, frozenset([30, 40]))
    print(hash(tuple_list_sequence))

    # 字典提供很多种构造函数， 字典推导式也可以构造字典
    # https://docs.python.org/3/library/stdtypes.html#mapping-types-dict
    dict_1 = dict(one=1, two=2, three=3)
    print(dict_1)
    
    dict_2 = {"one": 1, "two": 2, "three": 3}
    print(dict_2)

    dict_3 = dict(zip(["one", "two", "three"], [1, 2, 3]))
    print(dict_3)

    dict_4 = dict([("two", 2), ("three", 3), ("one", 1)])
    print(dict_4)

    dict_5 = dict({"three": 3, "one": 1, "two": 2})
    print(dict_5)

    print(dict_1 == dict_2 == dict_3 == dict_4 == dict_5)

    # 字典推导 dictcomp 可以从任何以键值对作为元素的可迭代对象中构建出字典
    DIAL_CODES = [
        (86, 'China'),
        (91, 'India'),
        (1, 'United States'),
        (62, 'Indonesia'),
        (55, 'Brazil'),
        (92, 'Pakistan'),
        (880, 'Bangladesh'),
        (234, 'Nigeria'),
        (7, 'Russia'),
        (81, 'Japan'),
    ]

    print()
    country_code = {country: code for code, country in DIAL_CODES}
    print(type(country_code))
    print(country_code, "\n")

    print(type(country_code.items()))
    print(country_code.items(), '\n')
    country_code_conditions = {code: country.upper() for country, code in country_code.items()}
    print(country_code_conditions)
    # 常见映射方法: dict; collections.defaultdict; collections.OrderedDict