import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('./data/HousingData.csv', header=0)
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

df_without_nan = df.dropna()

# 字段
# CRIM：城镇人均犯罪率
# ZN：住宅用地超过25000平方英尺的比例
# INDUS：城镇非零售营业面积占比
# CHAS：查尔斯河亚变量（如果临河有大片土地则为1；否则为0）
# NOX：一氧化氮浓度（千万分之一）
# RM：平均每户的房间数
# AGE：1940年以前建成的自用住房比例
# DIS：到五个波士顿就业中心的加权距离
# RAD：辐射可达的公路的指数
# TAX：每10,000美元的全额财产的税率
# PTRATIO：城镇师生比例
# B：1000人种非裔美国人的比例
# LSTAT：地位较低人口的百分比
# MEDV：自住房中位价（以千美元为单位）


print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 506 entries, 0 to 505
# Data columns (total 14 columns):
#  #   Column   Non-Null Count  Dtype
# ---  ------   --------------  -----
#  0   CRIM     486 non-null    float64
#  1   ZN       486 non-null    float64
#  2   INDUS    486 non-null    float64
#  3   CHAS     486 non-null    float64
#  4   NOX      506 non-null    float64
#  5   RM       506 non-null    float64
#  6   AGE      486 non-null    float64
#  7   DIS      506 non-null    float64
#  8   RAD      506 non-null    int64
#  9   TAX      506 non-null    int64
#  10  PTRATIO  506 non-null    float64
#  11  B        506 non-null    float64
#  12  LSTAT    486 non-null    float64
#  13  MEDV     506 non-null    float64
# dtypes: float64(12), int64(2)
# memory usage: 55.5 KB

print(df.describe())
#              CRIM          ZN       INDUS  ...           B       LSTAT        MEDV
# count  486.000000  486.000000  486.000000  ...  506.000000  486.000000  506.000000
# mean     3.611874   11.211934   11.083992  ...  356.674032   12.715432   22.532806
# std      8.720192   23.388876    6.835896  ...   91.294864    7.155871    9.197104
# min      0.006320    0.000000    0.460000  ...    0.320000    1.730000    5.000000
# 25%      0.081900    0.000000    5.190000  ...  375.377500    7.125000   17.025000
# 50%      0.253715    0.000000    9.690000  ...  391.440000   11.430000   21.200000
# 75%      3.560263   12.500000   18.100000  ...  396.225000   16.955000   25.000000
# max     88.976200  100.000000   27.740000  ...  396.900000   37.970000   50.000000
#
# [8 rows x 14 columns]

'''
    变量两两之间关系的散点图
'''
import seaborn as sns
import matplotlib.pyplot as plt
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
# cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# sns.pairplot(df[cols], height=2.5)
# plt.tight_layout()
# plt.show()


'''
    相关系数矩阵
'''
import numpy as np
# cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# # 相关性，越接近1 越正相关，越接近-1 越负相关
# # 需要首先把NAN去掉，否则coef算不出来
# cm = np.corrcoef(df_without_nan[cols].values.T)
# print(cm)
# #  cbar=True 表示显示颜色条，square=True 表示将热力图的宽高设置为相等
# # hm = sns.heatmap(cm)
# hm = sns.heatmap(cm, cbar=True, square=True, fmt='.2f', annot=True, annot_kws={'size':8}, yticklabels=cols, xticklabels=cols)
# plt.show()






'''
?????????????????

#重复值处理
# df = df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
plt.boxplot(df.iloc[:,0:1],showmeans=True,meanline=True)
plt.show()

# df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
# plt.figure(figsize=(15, 15))
# #对所有特征（收入中位数）画盒图（箱线图）
# for i in range(df.shape[1]):
#     plt.subplot(4,4,i+1)
#     plt.boxplot(df[:,i],showmeans = True ,meanline = True)
#     #x，y坐标轴标签
#     plt.xlabel(df['feature_names'][i])
# # plt.subplot(4,4,14)
# # #绘制直方图
# # plt.boxplot(y, showmeans = True ,meanline = True)
# # #x，y坐标轴标签
# # plt.xlabel('target')
# plt.show()
'''


# 变成rrl可用的形式
# df.to_csv("data_for_rrl/House.data", header = None, index = None)



# cols = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'RAD', 'TAX', 'PTRATIO', 'LSTAT', 'MEDV']
# df = df[cols]

# 只保留连续值
cols = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = df[cols]
df.to_csv("data_for_rrl/House_cont.data", header = None, index = None)

# df_without_nan.to_csv("data_for_rrl/House_no_nan.data", header = None, index = None)



