#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 4-2 Python 中的字节概念概述
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-21

bytes 或 bytearray 对象的各个元素是介于 0~255 (含)之间的整数, 一个字节 1byte=8bits
然而，二进制序列的切片始终是同一类型的二进制序列，包括长度为 1 的切片

结构体和内存视图
struct 模块提供了一些函数，把打包的字节序列转换成不同类型字段组成的元组，
还有一些函数用于执行反向转换，把元组转换成打包的字节序列。 
struct 模块能处理 bytes、bytearray 和 memoryview 对象。
memoryview 类不是用于创建或存储字节序列的，而是共享内存，
让你访问其他二进制序列、打包的数组和缓冲中的数据切片，而无需复制字节序列，
例如 Python Imaging Library (PIL) 就是这样处理图像的
"""

import array
import struct


# ----------------------------
if __name__ == "__main__":
    print("-------- bytes --------")
    cafe = bytes('café', encoding='utf_8')
    print(type(cafe))
    print(cafe)
    print(cafe[0])
    print(cafe[:1])

    # my_bytes[0] 获取的是一个整数，而 my_bytes[:1] 返回的是一个长度为 1 的bytes 对象
    # s[0] == s[:1] 只对 str 这个序列类型成立。str 类型的这个行为十分罕见
    # 对其他各个序列类型来说,s[i] 返回一个元素，而 s[i:i+1] 返回一个相同类型的序列,里面是 s[i] 元素

    print("-------- bytearray --------")
    cafe_array = bytearray(cafe)
    print(type(cafe_array))
    print(cafe_array)
    print(cafe_array[0])
    print(cafe_array[-1:])

    print("-------- bytes construction --------")
    print(bytes.fromhex('31 4B CE A9'))
    # 使用缓冲类对象构建二进制序列是一种低层操作，可能涉及类型转换
    numbers = array.array('h', [-2, -1, 0, 1, 2])
    print(bytes(numbers))

    # 结构体和内存视图 使用 memoryview 和 struct 查看一个 GIF 图像的首部
    # 结构体的格式： < 是小字节序， 3s3s 是两个 3 字节序列， HH 是两个 16 位二进制整数
    format_struct = '<3s3sHH'
    with open("./filter.gif", "rb") as file_ptr:
        img = memoryview(file_ptr.read())
    header = img[:10]
    print(bytes(header))
    print(struct.unpack(format_struct, header))
    # 删除引用，释放 memoryview 实例所占的内存
    del header
    del img

    # mmap — Memory-mapped file support: https://docs.python.org/3/library/mmap.html
    #  memoryview 和 struct 模块，如果要处理二进制数据: 
    # https://docs.python.org/3/library/stdtypes.html#memoryviews