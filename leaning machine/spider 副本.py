#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pycharm
@File    ：spider副本.py
@IDE     ：PyCharm
@Author  ：土拨鼠1024
@Date    ：2020/11/15 19:34
'''

import csv
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
import warnings
warnings.filterwarnings("ignore")


#导入股票数据
pro = ts.pro_api()
pro = ts.pro_api('d95ea8510cd48dcaf50422b0fa4e7312643245ba731fbd308f084385')
df = pro.daily(ts_code='600298.SH', start_date='20081201')
df = df.sort_index(ascending=False)#升序排序
df.to_csv('text', index=0)#数据存储为csv文件
print(df)
