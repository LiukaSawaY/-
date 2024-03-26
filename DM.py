import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

def loader(filepath):   #读取数据
    df = pd.read_csv(filepath, header=0)
    return df

def count(str,data): #输出标称数据频数
    return data[str].value_counts()

def fiveNumberandnull(str,data): #输出数值数据5数概括及缺失值的个数
    nums = data[str]

    # 缺失值的个数
    nullnum = nums.isnull().sum()
    print("缺省值null个数:%d"%(nullnum))

    # 五数概括 Minimum（最小值）、Q1、Median（中位数、）、Q3、Maximum（最大值）
    nums = nums.dropna(axis = 0) #删除NaN值
    Minimum = min(nums)
    Maximum = max(nums)
    Q1 = np.percentile(nums, 25)
    Median = np.median(nums)
    Q3 = np.percentile(nums, 75)
    print("Minimum:", Minimum)
    print("Q1:", Q1)
    print("Median:", Median)
    print("Q3:", Q3)
    print("Maximum:",Maximum)


def data_plot(numerical_attributes_list, df): # 绘制直方图和盒图
    for attribute in numerical_attributes_list:
        print(attribute, "的直方图：")
        plt.hist(df[attribute])
        plt.xlabel(attribute)
        plt.ylabel('Count')
        plt.title('histogram')
        plt.show()
        print(attribute, "的盒图：")
        plt.boxplot(df[attribute])
        plt.xlabel(attribute)
        plt.title('Box diagram')
        plt.show()

def null_fill_dele(data):
    print("将缺失部分剔除")
    data_dele = data.dropna()
    print("剔除缺失值前的数据集⼤⼩: ", data.shape)
    print("剔除缺失值后的数据集⼤⼩: ", data_dele.shape)
    print("剔除缺失值后缺失值统计：", data_dele.isnull().sum())
    return  data_dele

def null_fill_max(data, df, numerical_attributes_list):
    print("用最高频率值来填补缺失值")
    data_max_upda = data.fillna("JavaScript")
    print("替换缺失值前的数据集⼤⼩: ", data.shape)
    print("替换缺失值后的数据集⼤⼩: ", data_max_upda.shape)
    print("将缺失部分剔除后缺失值统计：", data_max_upda.isnull().sum(), "个")
    counts = data.value_counts()
    counts_max_upda = data_max_upda.value_counts()
    print("替换缺失值前频数统计\n", counts, "\n")
    print("替换缺失值后频数统计\n", counts_max_upda, "\n")
    return data_max_upda

def null_fill_rela(data, df, col):
    print("通过属性的相关关系来填补缺失值")
    print("相关属性：", col)
    mean = df[col].mean()
    data_rela_upda = data.fillna(mean)
    print("替换缺失值前的数据集⼤⼩: ", data.shape)
    print("替换缺失值后的数据集⼤⼩: ", data_rela_upda.shape)
    print("将缺失部分剔除后缺失值统计：", data_rela_upda.isnull().sum(), "个")
    counts = data.value_counts()
    counts_rela_upda = data_rela_upda.value_counts()
    print("替换缺失值前频数统计\n", counts, "\n")
    print("替换缺失值后频数统计\n", counts_rela_upda, "\n")
    return data_rela_upda


def null_fill_sim(df, numerical_attributes_list):
    print("通过数据对象之间的相似性来填补缺失值")
    df_num = df[numerical_attributes_list].astype('float64')
    knn_imputer = KNNImputer(n_neighbors=2)
    filled_values = knn_imputer.fit_transform(df_num)
    df_filled = pd.DataFrame(filled_values, columns=df.columns)

    # 打印填补后的DataFrame
    return df_filled

