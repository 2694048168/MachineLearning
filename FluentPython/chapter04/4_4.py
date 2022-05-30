#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 4-4 文本编解码和操作系统平台
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-22

如果脚本要生成文件，而字节的内容取决于平台或同一平台中的区域设置，那么就可能导致兼容问题。
需要在多台设备中或多种场合下运行的代码，一定不能依赖默认编码。
打开文件时始终应该明确传入 encoding= 参数，因为不同的设备使用的默认编码可能不同，有时隔一天也会发生变化。

编码默认值一团糟, 有几个设置对 Python I/O 的编码默认值有影响
"""

import os
import sys
import locale


# ----------------------------
if __name__ == "__main__":
    open("cafe.txt", "w", encoding="utf_8").write('café')
    text_sequence = open('cafe.txt').read()
    print(text_sequence, '\n') # output: caf茅

    # 分析在 Windows 操作系统平台下的默认编码问题
    file_ptr = open('cafe.txt', 'w', encoding='utf_8')
    print(file_ptr)
    file_ptr.write('café')
    file_ptr.close()
    print(os.stat('cafe.txt').st_size)

    file_ptr_2 = open('cafe.txt')
    print(file_ptr_2)
    print(file_ptr_2.encoding)
    print(file_ptr_2.read(), '\n')
    file_ptr_2.close()

    file_ptr_3 = open('cafe.txt', encoding='utf_8')
    print(file_ptr_3)
    print(file_ptr_3.read(), '\n')
    file_ptr_3.close()

    file_ptr_4 = open('cafe.txt', 'rb')
    print(file_ptr_4)
    print(file_ptr_4.read())
    file_ptr_4.close()

    # 编码默认值, Python I/O 的编码默认值
    print("--------- the default codec for Python I/O ---------")
    expressions = """
        locale.getpreferredencoding()
        type(my_file)
        my_file.encoding
        sys.stdout.isatty()
        sys.stdout.encoding
        sys.stdout.isatty()
        sys.stdin.encoding
        sys.stderr.isatty()
        sys.stderr.encoding
        sys.getdefaultencoding()
        sys.getfilesystemencoding()
    """
    my_file = open('dummy', 'w')
    for expression in expressions.split():
        value = eval(expression)
        print(expression.rjust(30), "->", repr(value))

# ----------------------------------------
# 在 Windows 10 操作系统平台上执行结果测试
"""
caf茅 

<_io.TextIOWrapper name='cafe.txt' mode='w' encoding='utf_8'>    
5
<_io.TextIOWrapper name='cafe.txt' mode='r' encoding='cp936'>    
cp936
caf茅

<_io.TextIOWrapper name='cafe.txt' mode='r' encoding='utf_8'>    
café

<_io.BufferedReader name='cafe.txt'>
b'caf\xc3\xa9'
--------- the default codec for Python I/O ---------
 locale.getpreferredencoding() -> 'cp936'
                 type(my_file) -> <class '_io.TextIOWrapper'>    
              my_file.encoding -> 'cp936'
           sys.stdout.isatty() -> True
           sys.stdout.encoding -> 'utf-8'
           sys.stdout.isatty() -> True
            sys.stdin.encoding -> 'utf-8'
           sys.stderr.isatty() -> True
           sys.stderr.encoding -> 'utf-8'
      sys.getdefaultencoding() -> 'utf-8'
   sys.getfilesystemencoding() -> 'utf-8'
"""

# ----------------------------------------
# 在 Ubuntu 22 操作系统平台上执行结果测试
"""
café 

<_io.TextIOWrapper name='cafe.txt' mode='w' encoding='utf_8'>
5
<_io.TextIOWrapper name='cafe.txt' mode='r' encoding='UTF-8'>
UTF-8
café 

<_io.TextIOWrapper name='cafe.txt' mode='r' encoding='utf_8'>
café 

<_io.BufferedReader name='cafe.txt'>
b'caf\xc3\xa9'
--------- the default codec for Python I/O ---------
 locale.getpreferredencoding() -> 'UTF-8'
                 type(my_file) -> <class '_io.TextIOWrapper'>
              my_file.encoding -> 'UTF-8'
           sys.stdout.isatty() -> True
           sys.stdout.encoding -> 'utf-8'
           sys.stdout.isatty() -> True
            sys.stdin.encoding -> 'utf-8'
           sys.stderr.isatty() -> True
           sys.stderr.encoding -> 'utf-8'
      sys.getdefaultencoding() -> 'utf-8'
   sys.getfilesystemencoding() -> 'utf-8'
"""