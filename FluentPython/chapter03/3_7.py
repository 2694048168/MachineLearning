#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 3-7 dict的实现(散列表)及其导致的结果
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-21
"""


# ----------------------------
if __name__ == "__main__":
    # 1. 键必须是可散列的
    # 2. 字典在内存上的开销巨大：散列表所耗费的空间
    # 3. 键查询很快
    # 4. 键的次序取决于添加顺序
    # ---- 将同样的数据以不同的顺序添加到 3 个字典里 ----
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
    dict_1 = dict(DIAL_CODES)
    print(dict_1.keys())

    dict_2 = dict(sorted(DIAL_CODES))
    print(dict_2.keys())

    dict_3 = dict(sorted(DIAL_CODES, key=lambda x:x[1]))
    print(dict_3.keys())

    assert dict_1 == dict_2 == dict_3

    # 5. 往字典里添加新键可能会改变已有键的顺序(散列表需要扩容)
    # 由此可知，不要对字典同时进行迭代和修改。
    # 如果想扫描并修改一个字典，最好分成两步来进行：
    # 首先对字典迭代，以得出需要添加的内容，把这些内容放在一个新字典里；
    # 迭代结束之后再对原有字典进行更新。
    # 在 Python 3 中， .keys()、 .items() 和 .values() 方法返回的都是字典视图。
    # 也就是说，这些方法返回的值更像集合，而不是像 Python 2 那样返回列表。
    # 视图还有动态的特性，它们可以实时反馈字典的变化。