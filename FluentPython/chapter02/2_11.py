#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 2-11 双向队列
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-21

利用 .append 和 .pop 方法，可以把列表当作栈或者队列来用
比如，把 .append 和 .pop(0) 合起来用，就能模拟队列的 “先进先出” 的特点
但是删除列表的第一个元素(抑或是在第一个元素之前添加一个元素)之类的操作是很耗时的，
因为这些操作会牵扯到移动列表里的所有元素。

collections.deque 类（双向队列）是一个线程安全、可以快速从两端添加或者删除元素的数据类型。
而且如果想要有一种数据类型来存放“最近用到的几个元素”， 
deque 也是一个很好的选择。
这是因为在新建一个双向队列的时候，你可以指定这个队列的大小，
如果这个队列满员了，还可以从反向端删除过期的元素，然后在尾端添加新的元素
"""

import collections


# ----------------------------
if __name__ == "__main__":
    deque_sequence = collections.deque(range(10), maxlen=10)
    print(deque_sequence)

    deque_sequence.rotate(3)
    print(deque_sequence)
    deque_sequence.rotate(-4)
    print(deque_sequence)

    deque_sequence.appendleft(-1)
    print(deque_sequence)

    deque_sequence.extend([11, 22, 33])
    print(deque_sequence)

    deque_sequence.extendleft([10, 20, 30, 40])
    print(deque_sequence)
    # append 和 popleft 都是原子操作，
    # 也就说是 deque 可以在多线程程序中安全地当作先进先出的队列使用，而使用者不需要担心资源锁的问题。
    # queue; multiprocessing; asyncio; heapq