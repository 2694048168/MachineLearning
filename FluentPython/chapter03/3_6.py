#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 3-6 集合 set and frozenset
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-21

集合中的元素必须是可散列的， set 类型本身是不可散列的，
但是 frozenset 可以。因此可以创建一个包含不同 frozenset 的 set
善用集合的(并交补)操作, 提高效率, 增加可读性
"""

import dis
from unicodedata import name

# ----------------------------
if __name__ == "__main__":
    list_sequence = ['spam', 'spam', 'eggs', 'spam']
    print(list_sequence)
    print(set(list_sequence))

    # 集合的并交补操作
    haystack = {"weili@123.com", "weili@124.com", "weili@125.com",
                "weili@113.com", "weili@114.com", "weili@115.com",
                "weili@133.com", "weili@134.com", "weili@135.com"}
    needles = {"weili@123.com", "weili@124.com", "weili@125.com"}

    haystack_tuple = ("weili@123.com", "weili@124.com", "weili@125.com",
                    "weili@113.com", "weili@114.com", "weili@115.com",
                    "weili@133.com", "weili@134.com", "weili@135.com")
    needles_tuple = ("weili@123.com", "weili@124.com", "weili@125.com")
    # 电子邮件地址的集合 haystack，
    # 还要维护一个较小的电子邮件地址集合 needles
    # 然后求出 needles 中有多少地址同时也出现在了 heystack 里
    print(type(haystack))
    found_num = len(needles & haystack)
    print(found_num)

    print(type(haystack_tuple))
    found_num = len(set(needles_tuple) & set(haystack_tuple))
    print(found_num)

    print(type(haystack_tuple))
    found_num = len(set(needles_tuple).intersection(haystack_tuple))
    print(found_num)

    found = 0
    for n in needles_tuple:
        if n in haystack_tuple:
            found += 1
    print(found_num)

    # set() 才是一个空集, {} 只是一个空字典
    # 集合字面量
    print(type(set()))
    print(type({}))

    set_sequence = {42}
    print(type(set_sequence))
    print(set_sequence.pop())
    print(set_sequence)

    # 两种不同的构造集合的方法
    dis.dis('{1}')
    dis.dis('set([1])')

    print(frozenset(range(10)))

    # 集合推导式 setcomps
    # 新建一个 Latin-1 字符集合，该集合里的每个字符的 Unicode 名字里都有 “SIGN” 这个单词
    set_comprehension = {chr(i) for i in range(32, 256) if 'SIGN' in name(chr(i),'')} 
    print(set_comprehension)