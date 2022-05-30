#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 4-3 Python 中编解码器 codec or encoder/decoder
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-21

每个编解码器都有一个名称，如 'utf_8'，而且经常有几个别名，如 'utf8'、 'utf-8' 和 'U8'
1. latin1 (即 iso8859_1)
2. cp1252
3. cp437
4. gb2312
5. utf-8
6. utf-16le

了解编解码问题
虽然有个一般性的 UnicodeError 异常，但是报告错误时几乎都会指明具体的异常：
UnicodeEncodeError (把字符串转换成二进制序列时)
UnicodeDecodeError (把二进制序列转换成字符串时)
如果源码的编码与预期不符，加载 Python 模块时还可能抛出 SyntaxError

# 如何找出字节序列的编码 (不能)
# https://pypi.org/project/chardet/

# BOM(即字节序标记 byte-order mark) 指明编码时使用 Intel CPU 的小字节序(小端序 little-endian)
"""


# ----------------------------
if __name__ == "__main__":
    print("-------- El Niño --------")
    for codec in ['latin_1', 'utf_8', 'utf_16', 'cp1252']:
        print(codec, "El Niño".encode(codec), sep="\t")

    print("-------- Wei Li --------")
    for codec in ['latin_1', 'utf_8', 'utf_16', 'cp1252', 'gb2312']:
        print(codec, "Wei Li".encode(codec), sep="\t")

    print("-------- 黎 为 --------")
    for codec in ['utf_8', 'utf_16', 'gb2312']:
        print(codec, "黎 为".encode(codec), sep="\t")

    # 1. 处理 UnicodeEncodeError 
    # https://docs.python.org/3/library/codecs.html#codecs.register_error
    print("-------- UnicodeEncodeError --------")
    city = 'São Paulo'
    print(city.encode('utf_8'))
    print(city.encode('utf_16'))
    print(city.encode('iso8859_1'))
    print(city.encode('cp437', errors="ignore"))
    print(city.encode('cp437', errors="replace"))
    print(city.encode('cp437', errors="xmlcharrefreplace"))
    # print(city.encode('cp437'))
    # print(city.encode('cp437', errors=strict))

    # 2. 处理 UnicodeDecodeError
    # 乱码字符称为鬼符（gremlin）或 mojibake
    print("-------- UnicodeDecodeError --------")
    octets = b'Montr\xe9al'
    print(octets.decode("cp1252"))
    print(octets.decode("iso8859_7"))
    print(octets.decode("koi8_r"))
    print(octets.decode("utf_8", errors="replace"))
    print(octets.decode("utf_8", errors="ignore"))
    # print(octets.decode("utf_8"))

    # 3. 处理 SyntaxError
    # 使用预期之外的编码加载模块时抛出的 SyntaxError
    print("-------- SyntaxError --------")