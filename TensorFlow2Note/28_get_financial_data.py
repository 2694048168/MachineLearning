#!/usr/bin/env python3
# encoding: utf-8

"""
@Filename: 28_get_financial_data.py
@Function: 利用 Tushare 获取金融时间序列数据
@Assignment: Financial Time Series Analysis
@Python Version: 3.8
@Author: Wei Li
@Date：2021-06
"""

import os

# http://tushare.org/
# Tushare 是一个免费、开源的 python 财经数据接口包
# pip install tushare
import tushare as ts
print("The Version of Tushare Currently: {}".format(ts.__version__))

def down_financial_data(datapath, code="600519", code_title="moutai", ktype="D", start="2010-01-01", end="2021-06-11"):
    print("------>>> Downloading ......")
    # 自己下载相关金融数据（数据点数不少于1000，即以日为单位，四年以上的数据）
    # 贵州茅台股票代码 code="600519"
    # 使用 Pro 版接口：https://waditu.com/document/2 该接口需要一些 Token 和权限
    # 本次使用原始接口 API 即可
    df_moutai = ts.get_k_data(code=code, ktype=ktype, start=start, end=end)

    print("------>>> Downloading Successfully!")

    print("------>>> Saving To File ......")
    # save data to a CSV file
    filename = os.path.join(datapath, f"{code_title}.csv")
    df_moutai.to_csv(filename)
    print(f"------>>> Saving To {filename} Successfully!")


# --------------进行本段代码功能测试----------------------
if __name__ == "__main__":
    # filepath to save financial data
    datapath = os.path.join(os.getcwd(), "csv_file")
    if not os.path.exists(datapath):
        os.makedirs(datapath)

    down_financial_data(datapath)