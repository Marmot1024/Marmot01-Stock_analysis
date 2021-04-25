import pandas as pd
from hdfs import InsecureClient


#连接到hdfs：bigdata01接口为50070
client_hdfs = InsecureClient('http://master:50070')



#读取hdfs数据
with client_hdfs.read('F:\IDEA\MJ_DATA1\A_list.csv', encoding='utf-8') as reader:
    df1 = pd.read_csv(reader,index_col=0)

#写入hdfs数据
with client_hdfs.write('F:\IDEA\MJ_DATA1\A_list.csv', encoding='utf-8',overwrite=True) as writer:
    df1.to_csv(writer)