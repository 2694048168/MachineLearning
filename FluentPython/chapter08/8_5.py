#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 8-5 弱引用
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-27

正是因为有引用，对象才会在内存中存在。
当对象的引用数量归零后，垃圾回收程序会把对象销毁。
但是，有时需要引用对象，而不让对象存在的时间超过所需时间。这经常用在缓存中。

弱引用不会增加对象的引用数量。
引用的目标对象称为所指对象 referent 。因此弱引用不会妨碍所指对象被当作垃圾回收。
弱引用在缓存应用中很有用，因为我们不想仅因为被缓存引用着而始终保存缓存对象。

# WeakValueDictionary
WeakValueDictionary 类实现的是一种可变映射，里面的值是对象的弱引用。
被引用的对象在程序中的其他地方被当作垃圾回收后，对应的键会自动从 WeakValueDictionary 中删除。
因此， WeakValueDictionary 经常用于缓存。

# 弱引用的局限
这些局限基本上是 CPython 的实现细节，在其他 Python 解释器中情况可能不一样
"""

import weakref


# ----------------------------
if __name__ == "__main__":
    # 弱引用是可调用的对象，返回的是被引用的对象；如果所指对象不存在了，返回 None
    set_seq_a = {0, 1}
    weak_reference = weakref.ref(set_seq_a)
    print(weak_reference)

    print(weak_reference())
    set_seq_a = {2, 3, 4}
    print(weak_reference())

    print(weak_reference() is None)
    print(weak_reference() is None)
    # https://docs.python.org/3/library/weakref.html