#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 2-8 list.sort 方法和内置函数 sorted
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-18
"""

import sys
import random
import bisect

# 用 bisect 来管理已排序的序列：bisect.bisect function and bisect.insort function
# 使用二分查找法来有序序列中查找或插入元素
# https://code.activestate.com/recipes/577197-sortedcollection/
HAYSTACK = [1, 4, 5, 6, 8, 12, 15, 20, 21, 23, 23, 26, 29, 30]
NEEDLES = [0, 1, 2, 5, 8, 10, 22, 23, 29, 30, 31]
ROW_FMT = "{0:2d} @ {1:2d} {2}{0:<2d}"

def demo(bisect_function):
    for needle in reversed(NEEDLES):
        position = bisect_function(HAYSTACK, needle)
        offset = position * '  |'
        print(ROW_FMT.format(needle, position, offset))

def grade(score, breakpoints=[60, 70, 80, 90], grades="FDCBA"):
    i = bisect.bisect(breakpoints, score)
    return grades[i]


# ----------------------------
if __name__ == "__main__":
    # list.sort 方法会就地排序列表，也就是说不会把原列表复制一份。
    # 这也是这个方法的返回值是 None 的原因，提醒你本方法不会新建一个列表。
    # 在这种情况下返回 None 其实是 Python 的一个惯例：
    # 如果一个函数或者方法对对象进行的是就地改动，那它就应该返回 None，
    # 好让调用者知道传入的参数发生了变动，而且并未产生新的对象。
    # 例如， random.shuffle 函数也遵守了这个惯例
    fruits = ['grape', 'raspberry', 'apple', 'banana']
    print(f"-------------before sorted---------\n{fruits}")
    # create a new object to return, non in-place
    print(f"-------------after  sorted---------\n{sorted(fruits)}")
    print(f"-------------after  sorted reverse=True---------\n{sorted(fruits, reverse=True)}")
    print(f"-------------after  sorted key=len---------\n{sorted(fruits, key=len)}")
    print(f"-------------after  sorted reverse=True key=len---------\n{sorted(fruits, reverse=True, key=len)}")

    print(f"\n-------------before sorted---------\n{fruits}")
    fruits.sort()  # 就地排序 in-place sort
    print(f"-------------after  sorted---------\n{fruits}")

    # ---------------------------------------------------------
    if sys.argv[-1] == "left":
        bisect_function = bisect.bisect_left
    else:
        bisect_function = bisect.bisect
        # bisect_function = bisect.bisect_right

    print(f"DEMO: {bisect_function.__name__}")
    print('haystack ->', ' '.join('%2d' % n for n in HAYSTACK))
    demo(bisect_function=bisect_function)

    # bisect 可以用来建立一个用数字作为索引的查询表格，比如说把分数和成绩对应起来
    grade_score = [grade(score) for score in [33, 99, 77, 70, 89, 90, 100]]
    print(f"\nthe grade for these score is :")
    print(grade_score)

    # 用 bisect.insort 插入新元素保持有序序列的有序性
    random.seed(42)
    list_sequence = []
    size_range = 7
    for idx in range(size_range):
        new_item = random.randrange(size_range*2)
        bisect.insort(list_sequence, new_item)
        print(f"{new_item:2d} -> {list_sequence}")