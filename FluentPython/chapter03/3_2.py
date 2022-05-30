#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 3-2 用 setdefault 处理找不到的建
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-21
"""

import sys
import re

# python 3_2.py zen_python.txt
# ----------------------------
if __name__ == "__main__":
    # 从索引中获取单词出现的频率信息，并把它们写进对应的列表里
    WORD_RE = re.compile(r'\w+')
    index_dict = {}
    with open(sys.argv[1], encoding='utf-8') as file_ptr:
        for line_num, line in enumerate(file_ptr, 1):
            for match in WORD_RE.finditer(line):
                word = match.group()
                column_num = match.start() + 1
                location = (line_num, column_num)
                # -------------------------------------------------
                # 1. 下面的实现不是一种好的方法
                # occurrences = index_dict.get(word, [])
                # occurrences.append(location)
                # index_dict[word] = occurrences
                # -------------------------------------------------
                # 2. 使用 setdefault 解决多次查询建的操作
                index_dict.setdefault(word, []).append(location)
                # -------------------------------------------------
    # 以字典顺序打印出结果
    for word in sorted(index_dict, key=str.upper):
        print(word, index_dict[word])