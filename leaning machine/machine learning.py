#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pycharm 
@File    ：machine learning.py
@IDE     ：PyCharm 
@Author  ：土拨鼠1024
@Date    ：2020/12/15 4:43 
'''

import csv
import pandas as pd
import tushare as ts
import numpy as np
from matplotlib.pyplot import MultipleLocator
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

df = pd.read_csv(r'text')
print(df.head(5))#查看头部5行数据

# # #读取csv文件
data = df
# file = pd.read_csv(r'text')


# 数据归一化处理 & 安装 keras和`tensorflow库
from sklearn import preprocessing as process
# 在数据分析之前先对所有的数据进行分析
# 后两项特征的数量级远大于其他项
X = data.loc[:, 'open':'amount'] #运用df.loc 筛选重组成新的dataframe, 'open':'amount'是执行切片操作，提取从‘open’列到‘amount’列
X = X.values #类型转换 df.values -> pandas


""""""""""""""""""""""""""""""""""""""""""""""""""""" 
绘图1
"""""""""""""""""""""""""""""""""""""""""""""""""""""
y = data["close"].values
# print(X.shape)#查看数据规模
# plt.plot(X.min(axis=0), 'v', label='min') # axis=0 or axis=1 分别对应的是坐标轴的纵，横，
# plt.plot(X.max(axis=0), '^', label='max') # 其中axis = 0 时，表示纵轴，方向从上到下
# plt.yscale('log') #scale 函数的作用是修改刻度，log就是他所要修改的刻度值，将轴的刻度变成了log（对数）
# plt.legend(loc='best', fontsize=14) #loc = best 是为了调整标签显示的位置
# plt.xlabel('features', fontsize=14) #fontsize是定义字体大小
# plt.ylabel('feature magnitude', fontsize=14)
# plt.show()



scaler = process.StandardScaler() #使用sklearn-preprocessing将数据标准化，去均值，将方差变成1
scaler.fit(X) #根据已有的训练数据创建一个标准化的转换器
X_scalerd = scaler.transform(X)

# 使用上面这个转换器去转换训练数据x,调用transform方法
y = pd.DataFrame(X_scalerd)[3].values #将X_scalerd中第四列的close 收盘价 进行转换 df -> pandas
temp_data = pd.DataFrame(X_scalerd)
temp_data = temp_data.iloc[-30:] #运用iloc函数 提取X_scaler中最近三十天的数据

""""""""""""""""""""""""""""""""""""""""""""""""""""" 
绘制最近三十天的收盘价
"""""""""""""""""""""""""""""""""""""""""""""""""""""
# plt.plot(temp_data[3], color='orange', label='Close Price')
# ax = plt.gca() #获取当前坐标对象
# x_major_locator =MultipleLocator(5)
# ax.xaxis.set_major_locator(x_major_locator) #x的取值只能为5的倍数
# plt.xticks(rotation=60) #角度设置
# plt.legend(loc='best')#loc = best 是为了调整标签显示的位置
# plt.show()

print(X_scalerd.shape, y.shape) #展示行列数


#重点` LSTM算法预测未来趋势

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
from keras.layers import RepeatVector
import keras

# 用t天的数据预测t+1天的，所以把y前移
# X有一个会多出来，所以删掉X的最后一个和y的第一个
import numpy as np

X_train = pd.DataFrame(X_scalerd)[[3, 5, 7]].values #提取出 close , change, vol 三列
X_train = np.delete(X_train, -1, axis=0) #删除x 的最后一个
y_train = np.delete(y, [1]) #删除y 的第一个
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1]) #运用reshape函数将其变成三维数组
y_train = y_train.reshape(y_train.shape[0], 1, 1) #同上
print(X_train.shape, y_train.shape) #查看行列数


model = Sequential() #调用sequential序贯模型
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dense(16, kernel_initializer="uniform", activation='relu'))
model.add(Dense(1, kernel_initializer="uniform", activation='linear'))
adam = keras.optimizers.Adam(decay=0.2)
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
model.summary()


# 训练模型
#准确率（accuracy），损失函数（loss function）
print(X_train.shape, y_train.shape)
history = model.fit(X_train, y_train, epochs=100, verbose=2, shuffle=False)
# model.save("1-1.h5")

# plot history 损失值和准确率绘图
# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['accuracy'], label='ac')
# plt.legend()
# plt.show()


#定义了一个预测涨跌是否正确的计算方法
def cal_ac_rate(ori, pre):
    ori_r = []
    pre_r = []
    ac = 0
    if ori.shape[0] != pre.shape[0]:
        return 0
    else:
        for i in range(0, ori.shape[0] - 1):
            if ori[i] - ori[i + 1] < 0:
                ori_r.append(1)
            if ori[i] - ori[i + 1] > 0:
                ori_r.append(-1)
            if ori[i] - ori[i + 1] == 0:
                ori_r.append(0)

            if pre[i] - pre[i + 1] < 0:
                pre_r.append(1)
            if pre[i] - pre[i + 1] > 0:
                pre_r.append(-1)
            if pre[i] - pre[i + 1] == 0:
                pre_r.append(0)
        for i in range(0, len(ori_r) - 1):
            if ori_r[i] == pre_r[i]:
                ac += 1

        return ac, len(ori_r)

predictes_stock_price = model.predict(X_train) #调用sequential序贯模型
predictes_stock_price = predictes_stock_price.reshape(predictes_stock_price.shape[0])
y_train = y_train.reshape(y_train.shape[0])
plt.plot(predictes_stock_price[-30:], label='pre', color='red')
plt.plot(y_train[-30:], label='ori', color='blue')
plt.legend()
plt.show()
cal_ac_rate(y_train, predictes_stock_price)

