import matplotlib.pyplot as plt
import pandas as pd
import re

number = {}
list = []
new_list = []
label = []
values = []

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def getdata():
    file = pd.read_csv('A股列表.csv', encoding='GBK')
    column = file[['所属行业']]
    column = column.values

    for i in column:
        data = re.split(" ", str(i[0]))
        list.append(data[-1])
        # print(data[-1])

    for i in list:
        if i not in new_list:
            new_list.append(i)

    for i in new_list:
        number[i] = list.count(i)

    print(number)
    return number
def show():

    for k, v in number.items():
        label.append(str(k))
        values.append(v)

    plt.bar(label, values,  align = 'center')
    plt.xticks(size='small', rotation=68,fontsize=13)
    plt.title('A股主板块行业分布图')#绘制标题

    plt.savefig('主版块行业分布图.png')#保存图片
    plt.show()

if __name__ == '__main__':
    getdata()
    show()
