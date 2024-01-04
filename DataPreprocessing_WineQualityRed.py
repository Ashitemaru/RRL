import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('./data/wineQuality/winequality-red.csv', header=0, sep = ';')
df_without_nan = df.dropna()




print(df.info())

print(df.describe())

# 变成rrl可用的形式
# df.to_csv("data_for_rrl/WineQuality_red.data", header = None, index = None)


# colnm = df.columns.to_list()
# fig = plt.figure(figsize=(15, 9))
# for i in range(12):
#     plt.subplot(3,4,i+1)  # 3行4列 位置是i+1的子图
#     df[colnm[i]].hist(bins=80)  # bins 指定显示多少竖条
#     plt.xlabel(colnm[i], fontsize=13)
#     plt.ylabel('Frequency')
# plt.tight_layout()
# plt.show()




'''
    相关系数矩阵
'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
# 相关性，越接近1 越正相关，越接近-1 越负相关
# 需要首先把NAN去掉，否则coef算不出来
cm = np.corrcoef(df_without_nan[cols].values.T)
print(cm)
#  cbar=True 表示显示颜色条，square=True 表示将热力图的宽高设置为相等
# hm = sns.heatmap(cm)
hm = sns.heatmap(cm, cbar=True, square=True, fmt='.2f', annot=True, annot_kws={'size':8}, yticklabels=cols, xticklabels=cols)
plt.show()

