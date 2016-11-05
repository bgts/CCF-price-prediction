# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline  
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures
import datetime
from statsmodels.tsa.arima_model import ARMA
#删掉原训练集、测试集文件的第一行
def readAsChunks(file_dir, types):
    chunks = []
    loop = True;
    chunk_size = 1000000
    reader = pd.read_csv(file_dir,',', header=None, iterator=True, dtype=types)
    # '+'号表示匹配前面的子表达式一次或多次
    # dtype参数，指明表中各列的类型，避免python自己猜，可以提高速度、并减少内存占用
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print('Iteration is stopped.')
            break
    df = pd.concat(chunks, ignore_index=True)
    #分块将.txt文件读入内存，放到一个 pandas 的 dataFrame 里。块大小（即每次读的行数）为chunk_size
    return df
def readAsDic(file_dir):
    dict_data = {}
    with open(file_dir, 'r') as df:
        for kv in [d.strip().split(',') for d in df]:
            if(dict_data.has_key((kv[1],kv[3]))==False):
                dict_data.setdefault((kv[1],kv[3]),[])
            dict_data[(kv[1],kv[3])].append(pd.to_datetime(kv[9],errors='coerce'))
    return dict_data
    
dic = readAsDic("product_market.csv")

df = readAsChunks("farming.csv", {8:np.float32, 9:np.float32,10:np.float32})#.replace("\N",np.nan)
#数据发布时间 对于格式有问题的时间设为NaT
df[12] = pd.to_datetime(df[12],errors='coerce')

#留下第0,1,2,3,9,12列
df.drop([4,5,6,7,8,10,11],axis=1,inplace=True)
#读测试数据
df_test = readAsChunks("product_market.csv",{})#.replace("null",np.nan)
#数据发布时间 对于格式有问题的时间设为NaT
df_test[9] = pd.to_datetime(df_test[9],errors='coerce')
#留下第0,1,2,3,9列
df_test.drop([4,5,6,7,8],axis=1,inplace=True)
#提取测试集中的市场名称映射值和农产品名称映射值
df_used = df_test.drop([0,2,9],axis=1,inplace=False)
df_used = df_used.drop_duplicates()
#提取根据测试集中的（市场名称映射值,农产品名称映射值），在训练集中提取相应的行
print 'ok'
df = pd.merge(df,df_used,on=[1,3])
df_used = []
x = []
y = []

print 'ok1'
df=df.sort_values(by=[1, 3],ascending=[1, 1])
df_test=df_test.sort_values(by=[1, 3],ascending=[1, 1])

#df.columns = [0, 1, 2, 3, 9, 12, 13]
key_tmp = ('02AAD134CD776815520A00CDC36A61E1','076095B1B9B448BF166FEB8A0EF80E83')
print 'ok2'
d1 = datetime.datetime(2015, 1, 1)
predict_time = pd.to_datetime(['2016-01-28'])
predict_time = (predict_time-d1).days%365
#print predict_time

for index, row in df.iterrows():   # 获取每行的index、row
    if key_tmp==(row[1],row[3]):
        x.append(pd.to_datetime(row[12]))
        y.append(row[9])
    else:
        #x = np.array(x)
        #y = np.array(y)
        print x
        print y
        y=pd.Series(y)
        y.index = pd.Index(x)
        start = datetime.datetime(2015,8,5)
        end = datetime.datetime(2016,1,21)
        print start
        arma =ARMA(y, order =(2,0))
        results = arma.fit()
        a = results.predict('2015-8-5','2016-1-21', dynamic=True)
        print a
        break
        '''
        print x
        print y
        clf = Pipeline([('poly', PolynomialFeatures(degree=7)),  
                    ('linear', LinearRegression(fit_intercept=False))])  
        clf.fit(x[:, np.newaxis], y)
        y_test = clf.predict(predict_time[:, np.newaxis])
        print y_test
        x = []
        y = []
        key_tmp = (row[1],row[3])
        predict_time = (pd.to_datetime(row[13])-d1).days%365
        print predict_time
        '''
