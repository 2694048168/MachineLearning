#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 5-5 函数注解
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-23

Python 3 提供了一种句法，用于为函数声明中的参数和返回值附加元数据。
注解不会做任何处理，只是存储在函数的 __annotations__ 属性 (一个字典)
Python 对注解所做的唯一的事情是，把它们存储在函数的 __annotations__ 属性里
仅此而已， Python 不做检查、不做强制、不做验证，什么操作都不做
换句话说, 注解对 Python 解释器没有任何意义
注解只是元数据，可以供 IDE、框架和装饰器等工具使用
函数注解的最大影响是为 IDE 和 lint 程序等工具中的静态类型检查功能提供额外的类型信息
"""

import inspect

# 有注解的 clip 函数
# 函数声明中的各个参数可以在 : 之后增加注解表达式。
# 如果参数有默认值，注解放在参数名和 = 号之间。
# 如果想注解返回值，在 ) 和函数声明末尾的 : 之间添加 -> 和一个表达式。
# 那个表达式可以是任何类型。
# 注解中最常用的类型是类（如 str 或 int）和字符串（如'int > 0'）
def clip(text:str, max_len:'int > 0'=80) -> str:
    """在max_len前面或后面的第一个空格处截断文本"""
    end = None
    if len(text) > max_len:
        space_before = text.rfind('', 0, max_len)
        if space_before >= 0:
            end = space_before
        else:
            space_after = text.rfind('', max_len)
        if space_after >= 0:
            end = space_after

    if end is None: # 没有找到空格
        end = len(text)
    return text[:end].rstrip()


# ----------------------------
if __name__ == "__main__":
    print(clip.__annotations__)

    # 从函数签名中提取注解
    signature_clip = inspect.signature(clip)
    print(signature_clip.return_annotation)
    for param in signature_clip.parameters.values():
        note = repr(param.annotation).ljust(13)
        print(note, ':', param.name, '=', param.default)