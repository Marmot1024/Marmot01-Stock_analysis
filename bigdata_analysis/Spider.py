#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pycharm 
@File    ：MySql create&&import.py
@IDE     ：PyCharm 
@Author  ：土拨鼠1024
@Date    ：2020/12/21 20:38 
'''
import tushare as ts

Stock_list = []
for i in range(1, 1980): #1 to 1979
    i = str(i)
    i = i.zfill(6)
    Stock_list.append(i)

for num in Stock_list:
    csv_name = num + '.csv'
    code_name = num + '.SZ'
    pro = ts.pro_api()
    pro = ts.pro_api('d95ea8510cd48dcaf50422b0fa4e7312643245ba731fbd308f084385')
    df = pro.daily(ts_code=code_name, start_date='20081201')
    df = df.sort_index(ascending=False)  # 升序排序
    if df.empty == True:
        continue
    df.to_csv(csv_name, index=0)  # 数据存储为csv文件
    print(num)



