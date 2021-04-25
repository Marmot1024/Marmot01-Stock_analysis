import pandas as pd
import csv
import os
import re

file = pd.read_csv("A股列表.csv", encoding='GBK', thousands=',')
file = pd.read_csv('A_list1.csv', thousands=',')
# print(file.head(5))
file = file[['板块', '公司全称', '注册地址', 'A股代码', 'A股简称', 'A股上市日期', 'A股总股本', 'A股流通股本', '地      区', '省    份', '城     市', '所属行业']]

column = file['所属行业']

new_column = []
for i in column:
    res = ''.join(re.findall('[\u4e00-\u9fa5]', i))
    new_column.append(res)

file['所属行业'] = pd.DataFrame(new_column)


# print(new_data.head(5))
file.to_csv('A_list.csv', index=0, encoding='utf-8')
