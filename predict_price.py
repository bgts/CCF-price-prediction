# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline  
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures
import datetime
from statsmodels.tsa.arima_model import ARMA
from scipy import interpolate
from old_TS import time_sequence_predict
import decimal
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
'''
def readAsDic(file_dir):
    dict_data = {}
    with open(file_dir, 'r') as df:
        for kv in [d.strip().split(',') for d in df]:
            if(dict_data.has_key((kv[1],kv[3]))==False):
                dict_data.setdefault((kv[1],kv[3]),[])
            dict_data[(kv[1],kv[3])].append(pd.to_datetime(kv[9],errors='coerce'))
    return dict_data
    
dic = readAsDic("product_market.csv")#读入测试集文件 将测试集存成字典 key:(市场映射值,产品映射值) value:待预测的日期list
'''
df = readAsChunks("farming.csv", {8:np.float32, 9:np.float32,10:np.float32})
#数据发布时间 对于格式有问题的时间设为NaT
df[12] = pd.to_datetime(df[12],errors='coerce')

#留下训练集的第0,1,3,9,12列
df.drop([4,5,6,7,8,10,11],axis=1,inplace=True)
#读测试数据
df_test = readAsChunks("product_market.csv",{})
#数据发布时间 对于格式有问题的时间设为NaT
df_test[4] = pd.to_datetime(df_test[4],errors='coerce')

#提取测试集中的市场名称映射值和农产品名称映射值
df_test = df_test.drop([0,4],axis=1,inplace=False)
df_test = df_test.drop_duplicates()
#根据测试集中的（市场名称映射值,农产品名称映射值），在训练集中提取相应的行
df = pd.merge(df,df_test,how='inner',on=[1,2,3])



#对训练集数据按（市场名称映射值,种类,农产品名称映射值）排序
df=df.sort_values(by=[1, 3, 2],ascending=[1, 1, 1])
#对测试集数据按（市场名称映射值,农产品名称映射值）排序
#df_test=df_test.sort_values(by=[1, 3],ascending=[1, 1])

#初始化
x = []#存日期
y = []#存农产品价格
key_tmp = ('02AAD134CD776815520A00CDC36A61E1','蔬菜','076095B1B9B448BF166FEB8A0EF80E83')
#排序后训练集中的第一个（市场名称映射值,农产品名称映射值）二元组
#key_tmp存（市场名称映射值,农产品名称映射值）二元组
last_date = datetime.datetime(2016,6,30)
predict_dates = pd.date_range(datetime.datetime(2016,7,1), periods=31).tolist()
file_object = open('result.csv','w')
count=0
count_total=0
for index, row in df.iterrows():   # 获取训练集中每行的索引和内容
    if key_tmp!=(row[1],row[2],row[3]):#比较当前行的第1列和第3列组成的二元组是否和key_tmp一致
        #用线性插值补全数据
        data = [(score, name) for score, name in zip(x,y)]
        data.sort() 
        x=[score for score,name in data] #将排好序的分数姓名的元组分开
        y=[name for score,name in data]
        base = min(x)
        ceiling = max(x)
        x_full_dates = pd.date_range(base, periods=(ceiling-base).days+1, freq='D').tolist()
        x_interpld = [(i-base).days for i in x]
        x_interpld = np.array(x_interpld)
        y = np.array(y)
        s1rev = interpolate.CubicSpline (x_interpld, y)
        #s1rev = interpolate.UnivariateSpline (x_interpld, y)
        #s1rev = interpolate.InterpolatedUnivariateSpline(x_interpld, y)
        #s1rev = interpolate.Akima1DInterpolator(x_interpld, y)
        #s1rev = interpolate.PchipInterpolator(x_interpld, y)
        #s1rev = interpolate.interp1d(x_interpld, y)
        
        #tck = interpolate.splrep(x_interpld, y)
        x_full_days = [(i-base).days for i in x_full_dates]
        x_full_days = np.array(x_full_days)
        y_interp = s1rev(x_full_days)
        #y_interp = interpolate.splev(x_full_days, tck)
        ceiling = max(x_full_dates)
        if last_date != ceiling:
            x_compensate = pd.date_range(ceiling + datetime.timedelta(days=1), periods=(last_date-ceiling).days, freq='D').tolist()
            x_full_dates.extend(x_compensate)
            y_compensate = [y_interp[-1] for i in range(len(x_compensate))]
            y_interp = y_interp.tolist()
            y_interp.extend(y_compensate)
        x = x_full_dates
        y = y_interp
        y = pd.Series(y)#将list类型转换成series，不然做不了时间序列
        y.index = pd.Index(x)#将日期作为农产品价格的索引
        #print y
        try:
            predict_series = time_sequence_predict(y)
            y=y.append(predict_series)
        
        except Exception,e:  #若时间序列有错 用滚动平均值预测
            print Exception,":",e
            count+=1
            i = 6
            sum_price = 0
            for j in range(1,i+1)[::-1]:
                try:
                    sum_price += y[-j]
                except:
                    i -= 1
                    continue
            a = sum_price/i
            for date in predict_dates:
                s = pd.Series([a], index=[date])
                y=y.append(s)
                a = (sum_price - y[-i-1] + a)/i                
        
        for date in predict_dates:
            splt = str(date).split(' ')
            string = str(key_tmp[0])+","+str(key_tmp[1])+","+str(key_tmp[2])+","+str(splt[0])+","+str(round(decimal.Decimal(y[date]*10))/10)+'\n'
            file_object.write(string)
        x = []
        y = []
        key_tmp = (row[1],row[2],row[3])
        count_total+=1
        #break
    x.append(pd.to_datetime(row[12]))#将当前行的第12列(日期)加入x
    y.append(row[9])#将当前行的第9列(农产品价格)加入x
data = [(score, name) for score, name in zip(x,y)]
data.sort()  
x=[score for score,name in data] #将排好序的分数姓名的元组分开
y=[name for score,name in data]
try:
    a=(y[-5]+y[-4]+y[-3]+y[-2]+y[-1])/5
except:
    print key_tmp
    
try:
    for date in predict_dates:
        splt = str(date).split(' ')
        string = str(key_tmp[0])+","+str(key_tmp[1])+","+str(key_tmp[2])+","+str(splt[0])+","+str(round(decimal.Decimal(a*10))/10)+'\n'
        file_object.write(string)
except:
    file_object.close()
file_object.close()
print count,count_total
