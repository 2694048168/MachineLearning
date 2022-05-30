#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 5-4 从定位参数到仅限关键字参数
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-23

Python 最好的特性之一是提供了极为灵活的参数处理机制，
而且 Python 3 进一步提供了仅限关键字参数(keyword-only argument)
与之密切相关的是，调用函数时使用 * 和 ** “展开”可迭代对象，映射到单个参数。

# 获取关于参数的信息
函数对象有个 __defaults__ 属性，它的值是一个元组，
里面保存着定位参数和关键字参数的默认值。
仅限关键字参数的默认值在 __kwdefaults__ 属性中。
然而,参数的名称在__code__ 属性中，它的值是一个 code 对象引用，自身也有很多属性。

inspect 模块的帮助下,Python 数据模型把实参绑定给函数调用中的形参的机制，这与解释器使用的机制相同
"""

import inspect


def tag(name, *content, cls=None, **attrs):
    """生成一个或多个HTML标签"""
    if cls is not None:
        attrs["class"] = cls

    if attrs:
        attr_str = ''.join(' %s="%s"' % (attr, value) for attr, value in sorted(attrs.items()))
    else:
        attr_str = ''

    if content:
        return '\n'.join('<%s%s>%s</%s>' % (name, attr_str, c, name) for c in content)
    else:
        return '<%s%s />' % (name, attr_str)

def clip(text, max_len=80):
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
    # tag function call-way
    print(tag('br'))
    print(tag('p', 'hello'))
    print(tag('p', 'hello', 'world'))
    print(tag('p', 'hello', id=33))
    print(tag('p', 'hello', 'world', cls='sidebar'))
    print(tag(content='testing', name='img'))
    my_tag = {'name': 'img', 'title': 'Sunset Boulevard', 'src': 'sunset.jpg', 'cls': 'framed'}
    print(tag(**my_tag))

    # 提取关于函数参数的信息
    print('---------------------------')
    print(clip.__defaults__)
    print(clip.__code__)
    print(clip.__code__.co_varnames)
    print(clip.__code__.co_argcount)

    # 使用 inspect 模块 提取函数的签名
    print('---------------------------')
    signature_function = inspect.signature(clip)
    print(signature_function)
    print(str(signature_function))
    for name, param in signature_function.parameters.items():
        print(param.kind, ":", name, "=", param.default)

    # 把 tag 函数的签名绑定到一个参数字典上
    print('---------------------------')
    sig = inspect.signature(tag)
    bound_args = sig.bind(**my_tag)
    print(bound_args)
    for name, value in bound_args.arguments.items():
        print(name, "=", value)

    del my_tag['name']
    # bound_args = sig.bind(**my_tag)