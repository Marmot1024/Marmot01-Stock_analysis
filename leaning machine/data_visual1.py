#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pycharm 
@File    ：data_visual1.py
@IDE     ：PyCharm 
@Author  ：土拨鼠1024
@Date    ：2020/11/30 19:12 
'''

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates

style.use('ggplot')

df = pd.read_csv('text1', parse_dates=True, index_col=0) #parse_dates=True 是图像带日期的关键

df_ohlc = df['close'].resample('10D').ohlc()
df_volume = df['volume'].resample('10D').sum()

print(df_ohlc.head())

df_ohlc.reset_index(inplace=True)
df_ohlc['date'] = df_ohlc['date'].map(mdates.date2num)

fig = plt.figure()
ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
plt.show()

