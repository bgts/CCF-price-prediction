# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline  
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures
import datetime
from scipy.interpolate import interp1d
'''删掉原训练集、测试集文件的第一行'''
def rmse(y_test, y):''''' 均方误差根 '''    
    return sp.sqrt(sp.mean((y_test - y) ** 2))  
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
    
#dic = readAsDic("product_market.csv")

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
x = {}
y = {}

print 'ok1'
df=df.sort_values(by=[1, 3],ascending=[1, 1])
df_test=df_test.sort_values(by=[1, 3],ascending=[1, 1])

#df.columns = [0, 1, 2, 3, 9, 12, 13]
key_tmp = ('02AAD134CD776815520A00CDC36A61E1','076095B1B9B448BF166FEB8A0EF80E83')
print 'ok2'
d1 = datetime.datetime(2005, 11, 26)
#predict_time = pd.to_datetime(['2015-12-31'])
#predict_time = (predict_time-d1).days
#print predict_time
count = 0
count_sum = 0 
for index, row in df.iterrows():   # 获取每行的index、row
    count_sum += 1
    if key_tmp!=(row[1],row[3]):
        #linear_interp = interp1d(x, y)
        rmse_list = []
        model_list = []
        for key,value in x.items():#字典x中，key是年份，value是训练集中出现的key年11月到(key+1)年2月的价格
            if key < 2015:#2015年的数据另作处理
                base = min(value)
                print min(value),max(value)
                #daylist_full = [i for i in range((max(value)-base).days+1)]
                #datelist_full = pd.date_range(base, periods=(max(value)-base).days).tolist()#生成从key年11月到(key+1)年2月内120天的所有日期
                x[key] = [(i-base).days for i in x[key]]
                #interpld_x = list(set(daylist_full).difference(set(x[key]))) #求差集，即待插值的x
                interpld_x = [i for i in range((max(value)-base).days+1)]
                #用线性插值补全数据
                linear_interp = interp1d(x[key], y[key])
                interpld_y = linear_interp(interpld_x)  
                plt.scatter(x[key], y[key], s=50) 
                plt.plot(interpld_x, interpld_y,'-') 
                plt.grid()  
                plt.show()  
                interpld_x = np.array(interpld_x)
                interpld_y = np.array(interpld_y)
                model = Pipeline([('poly', PolynomialFeatures(degree=7)),  
                                ('linear', LinearRegression(fit_intercept=False))])  
                print model
                model.fit(interpld_x[:, np.newaxis], interpld_y)
                predict_y = model.predict(interpld_x[:, np.newaxis])
                rmse_list.append(rmse(predict_y, interpld_y))
                model_list.apend(model)
            else:
                best_model = model_list[rmse_list.index(min(rmse_list))]
                x = np.array(x)
                y = np.array(y)
        break
        x = {}
        y = {}
        key_tmp = (row[1],row[3])
        
    tmp_date = pd.to_datetime(row[12])
    tmp_year = 0
    if tmp_date.month >= 11:
        tmp_year = tmp_date.year
        #x.append((tmp_date-d1).days)
    elif tmp_date.month <= 2:
        tmp_year = tmp_date.year-1
    if(x.has_key(tmp_year)==False):
        x.setdefault(tmp_date.year,[])
        y.setdefault(tmp_date.year,[])
    if tmp_date.month >= 11 or tmp_date.month <= 2:
        x[tmp_year].append(tmp_date)
        y[tmp_year].append(row[9])
    if tmp_date.year < 2010:
        count += 1
        
        
        
        '''
        x = np.array(x)
        y = np.array(y)
        plt.scatter(x, y, s=50)  
        model = Pipeline([('poly', PolynomialFeatures(degree=7)),  
                    ('linear', LinearRegression(fit_intercept=False))])  
        model.fit(x[:, np.newaxis], y)
        y_predict = model.predict(predict_time[:, np.newaxis])
        y_test = model.predict(x[:, np.newaxis])
        plt.plot(x, y_test,'*')  
        print y_predict
        plt.grid()  
        plt.show()  
        break
        '''
        
        #predict_time = (pd.to_datetime(row[13])-d1).days
        #print predict_time
print count_sum,count
