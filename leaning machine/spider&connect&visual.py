#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pycharm
@File    ：spider&connect&visual.py
@IDE     ：PyCharm
@Author  ：土拨鼠1024
@Date    ：2020/11/15 19:34
'''

import csv
import pandas as pd
import tushare as ts #tushare是个开源的python数据接口
import numpy as np

from sqlalchemy import create_engine
# import MySQLdb

#数据库中如果已经有housing price此表，则删除已存在的此表
# cursor.execute("DROP TABLE IF EXISTS housing price")

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
import warnings
warnings.filterwarnings("ignore")

#导入股票数据
df = ts.get_hist_data('600298', start='2019,1,1')
df = df.sort_index(ascending=True)#升序排序
df.to_csv('text1')#数据存储为csv文件
print(df)

#提取收盘价格构成新的Dateframe, 选的是收盘价和平均价格组成的新df
sample = {'收盘价': df.close, '平均价': (df.high+df.low)/2.0}
# sample = {'close': df.close}
date = pd.DataFrame(sample)

#收盘价格可视化 设置日期坐标为重难点
date.plot(kind='line')
plt.xticks(rotation='25')
plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""" 
将爬取到的数据集保存到mysql数据库中
"""""""""""""""""""""""""""""""""""""""""""""""""""""
#连接导入数据库
yconnect = create_engine('mysql+mysqldb://root:lqy720914@localhost:3306/stock_analysis?charset=utf8')
pd.io.sql.to_sql(df, 'stock_data', yconnect, schema='stock_analysis', if_exists='append')

