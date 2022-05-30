#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 示例 15-2 用于原地重写文件的上下文管理器
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-05-30

# contextlib 模块中的实用工具
https://docs.python.org/3/library/contextlib.html
"""

"""
import csv

# ----------------------------
if __name__ == "__main__":
    with inplace(csvfilename, 'r', newline='') as (infh, outfh):
        reader = csv.reader(infh)
        writer = csv.writer(outfh)

        for row in reader:
            row += ["new", "columns"]
            writer.writerow(row)
"""