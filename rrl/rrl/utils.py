import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            # print("tokens: "+str(tokens))
            f_list.append(tokens)
        # print("f_list: "+str(f_list))
    return f_list[:-1], int(f_list[-1][-1])


def read_csv(data_path, info_path, shuffle=False):
    D = pd.read_csv(data_path, header=None)
    if shuffle:
        D = D.sample(frac=1, random_state=0).reset_index(drop=True)
    f_list, label_pos = read_info(info_path)
    f_df = pd.DataFrame(f_list)
    print(f_df)
    print("D.columns: "+str(D.columns))
    D.columns = f_df.iloc[:, 0]
    print(D.columns)
    y_df = D.iloc[:, [label_pos]]
    X_df = D.drop(D.columns[label_pos], axis=1)
    f_df = f_df.drop(f_df.index[label_pos])
    return X_df, y_df, f_df, label_pos


class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, discrete=False, y_one_hot=True, drop='first'):
        self.f_df = f_df
        self.discrete = discrete
        self.y_one_hot = y_one_hot
        self.label_enc = preprocessing.OneHotEncoder(categories='auto') if y_one_hot else preprocessing.LabelEncoder()
        self.feature_enc = preprocessing.OneHotEncoder(categories='auto', drop=drop)
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.label_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.X_fname = None
        self.y_fname = None
        self.discrete_flen = None
        self.continuous_flen = None
        self.mean = None
        self.std = None

    def split_data(self, X_df):
        discrete_data = X_df[self.f_df.loc[self.f_df[1] == 'discrete', 0]]
        continuous_data = X_df[self.f_df.loc[self.f_df[1] == 'continuous', 0]]
        if not continuous_data.empty:
            continuous_data = continuous_data.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            continuous_data = continuous_data.astype(np.float)
        return discrete_data, continuous_data

    def fit(self, X_df, y_df):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        # 这里也是，本来是对分类的处理，要改成连续值的
        # self.label_enc.fit(y_df)
        # self.y_fname = list(self.label_enc.get_feature_names_out(y_df.columns)) if self.y_one_hot else y_df.columns
        self.label_imp.fit(y_df.values)
        self.y_fname = y_df.columns

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if do not discretize them.
            self.imp.fit(continuous_data.values)
        if not discrete_data.empty:
            # One-hot encoding
            self.feature_enc.fit(discrete_data)
            feature_names = discrete_data.columns
            self.X_fname = list(self.feature_enc.get_feature_names_out(feature_names))
            self.discrete_flen = len(self.X_fname)
            if not self.discrete:
                self.X_fname.extend(continuous_data.columns)
        else:
            self.X_fname = continuous_data.columns
            self.discrete_flen = 0
        self.continuous_flen = continuous_data.shape[1]

    def transform(self, X_df, y_df, normalized=False, keep_stat=False):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        # print("y_df--------: "+str(y_df))
        discrete_data, continuous_data = self.split_data(X_df)
        # Encode string value to int index.
        # print("y_df.values.reshape(-1, 1)--------: "+str(y_df.values.reshape(-1, 1)))

        # 这里用label_enc.transform是用来处理离散值的！！！也就是分类任务的标签
        # 现在变成连续值了，应该按照continuous_data来处理y_df这一列
        # y = self.label_enc.transform(y_df.values.reshape(-1, 1))

        y_df = y_df.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
        y_df = y_df.astype(np.float)

        # print("continuous_data: "+str(continuous_data.shape))
        # print("y_df: " + str(y_df.shape))


        y = pd.DataFrame(self.label_imp.transform(y_df.values), columns=y_df.columns)
        # y = y_df


        # label不用归一化
        # 这里先归一化处理，最后算mse的时候在反归一化
        # print("y_mean: "+str(y.mean()))
        # print("y_std: "+str(y.std()))
        # if normalized:
        #     if keep_stat:
        #         self.mean = y.mean()
        #         self.std = y.std()
        #     y = (y - self.mean) / self.std

        # print("y1--------: "+str(y))
        # if self.y_one_hot:
        #     y = y.toarray()
        # print("y2--------: "+str(y))

        # print(continuous_data.empty)
        # print(discrete_data.empty)
        # continuous_data and discrete_data 是dataframe
        if not continuous_data.empty:
            # print("here_please")
            # Use mean as missing value for continuous columns if we do not discretize them.
            continuous_data = pd.DataFrame(self.imp.transform(continuous_data.values),
                                           columns=continuous_data.columns)
            if normalized:
                if keep_stat:
                    self.mean = continuous_data.mean()
                    self.std = continuous_data.std()
                continuous_data = (continuous_data - self.mean) / self.std
            # print("nan not in continuous_data")
            # if continuous_data.isnull().values.any() == True:
            #     print("nan in continuous data!!!!!!!!!!!!!")
            # else:
            #     print("nan not in continuous_data")

        if not discrete_data.empty:
            # One-hot encoding
            if discrete_data.isnull().values.any() == True:
                print("nan in discrete data!!!!!!!!!!!!!")
            discrete_data = self.feature_enc.transform(discrete_data)

            if not self.discrete:
                X_df = pd.concat([pd.DataFrame(discrete_data.toarray()), continuous_data], axis=1)
            else:
                X_df = pd.DataFrame(discrete_data.toarray())

            # print(discrete_data.type)


        else:
            X_df = continuous_data


        # 它对离散值的处理会出现nan啊！！！！！！！！
        # 先不管离散值试试，看梯度还会不会nan
        # return continuous_data.values, y.values #这样不行。。。。。。。
        return X_df.values, y.values




