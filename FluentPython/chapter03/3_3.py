#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 3-3 映射(字典)的弹性键查询
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

import sys
import re
import collections

# ----------------------------
if __name__ == "__main__":
    # 从索引中获取单词出现的频率信息，并把它们写进对应的列表里
    WORD_RE = re.compile(r'\w+')
    # 创建 defaultdict 的时候没有指定 default_factory，查询不存在的键会触发 KeyError
    # index_dict = collections.defaultdict()
    index_dict = collections.defaultdict(list)
    with open(sys.argv[1], encoding='utf-8') as file_ptr:
        for line_num, line in enumerate(file_ptr, 1):
            for match in WORD_RE.finditer(line):
                word = match.group()
                column_num = match.start() + 1
                location = (line_num, column_num)
                # -------------------------------------------------------------
                # defaultdict 里的 default_factory 只会在 __getitem__ 里被调用
                # 特殊方法 __missing__ 它会在 defaultdict 遇到找不到的键的时候调用 default_factory
                index_dict[word].append(location)
                # -------------------------------------------------------------
    # 以字典顺序打印出结果
    for word in sorted(index_dict, key=str.upper):
        print(word, index_dict[word])