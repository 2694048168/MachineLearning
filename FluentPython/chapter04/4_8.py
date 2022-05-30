#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 4-8 Unicode 文本排序
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-22
"""

import locale
import unicodedata
import re

# pip install pyuca 
# https://pypi.org/project/pyuca/
import pyuca


# ----------------------------
if __name__ == "__main__":
   fruits = ['caju', 'atemoia', 'cajá', 'açaí', 'acerola']
   print(fruits)
   print(sorted(fruits))

   print(locale.setlocale(locale.LC_COLLATE, 'pt_BR.UTF-8'))
   fruits = ['caju', 'atemoia', 'cajá', 'açaí', 'acerola']
   print(fruits)
   print(sorted(fruits))

   coll = pyuca.Collator()
   sorted_fruits = sorted(fruits, key=coll.sort_key)
   print(fruits)
   print(sorted_fruits)

   # Unicode 数据库中数值字符的元数据示例（各个标号说明输出中的各列）
   print("-------------------------------------------------------")
   re_digit = re.compile(r"\d")
   sample = '1\xbc\xb2\u0969\u136b\u216b\u2466\u2480\u3285'
   for char in sample:
      print('U+%04x' % ord(char),
            char.center(6),
            're_dig' if re_digit.match(char) else '-',
            'isdig' if char.isdigit() else '-',
            'isnum' if char.isnumeric() else '-',
            format(unicodedata.numeric(char), '5.2f'),
            unicodedata.name(char),
            sep='\t')