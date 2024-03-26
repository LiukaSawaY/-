# 数据挖掘 第三周互评作业

## 1120202696 刘思远 计算机学院

### 1. 问题描述

本次作业中，自行选择2个数据集进行探索性分析与预处理。

### 2. 数据集

本次选用数据集为：

·GitHub Dataset

·Alzheimer Disease and Healthy Aging Data In US

### 3. 数据分析要求

#### 3.1 数据摘要和可视化

##### 数据摘要

标称属性，给出每个可能取值的频数

​      

##### 数据可视化

使用直方图、盒图等检查数据分布及离群点

#### 3.2 数据缺失的处理

观察数据集中缺失数据，分析其缺失的原因。分别使用下列四种策略对缺失值进行处理:

将缺失部分剔除

用最高频率值来填补缺失值

通过属性的相关关系来填补缺失值

通过数据对象之间的相似性来填补缺失值

注意：在处理后完成，要对比新旧数据集的差异。

### 4.提交内容

分析过程报告（PDF格式）

程序所在代码仓库地址（使用Github或码云），仓库中应包含完整的处理数据的代码和使用说明

所选择的数据集在仓库的README文件中说明

相关的数据文件不要上传到代码仓库中

建议：使用Jupyter Notebook将分析报告和代码组织在一起，使用Notebook的导出功能将报告导出为PDF格式的文件上传到乐学。

### 一、对GitHub Dataset的数据处理

#### 数据摘要

标称属性： 标称属性，给出每个可能取值的频数，包括repositories项目仓库名称、language编程语言


```python
import numpy as np
import pandas as pd
from pandas import DataFrame

def loader(filepath):   #读取数据
    df = pd.read_csv(filepath, header=0)
    return df

def count(str,data): #输出标称数据频数
    return data[str].value_counts()

df = loader("./Github Dataset/github_dataset.csv")
nominal_attribute_list = ["repositories", "language"]

for attribute in nominal_attribute_list:
    print(attribute, "的频数：")
    print(count(attribute, df), "\n")
```

    repositories 的频数：
    repositories
    kameshsampath/ansible-role-rosa-demos         2
    aloisdeniel/bluff                             2
    antoniaandreou/github-slideshow               2
    jgthms/bulma-start                            2
    artkirienko/hlds-docker-dproto                2
                                                 ..
    WhiteHouse/CIOmanagement                      1
    0xCaso/defillama-telegram-bot                 1
    ethereum/blake2b-py                           1
    openfoodfacts/folksonomy_mobile_experiment    1
    gamemann/All_PropHealth                       1
    Name: count, Length: 972, dtype: int64 
    
    language 的频数：
    language
    JavaScript          253
    Python              155
    HTML                 72
    Java                 44
    CSS                  37
    TypeScript           37
    Dart                 36
    C++                  29
    Jupyter Notebook     29
    Ruby                 28
    C                    26
    Shell                25
    PHP                  16
    Go                   15
    Rust                 10
    Swift                10
    C#                    8
    Objective-C           8
    Kotlin                7
    Makefile              6
    Jinja                 5
    SCSS                  4
    CoffeeScript          3
    Perl                  3
    Dockerfile            3
    Solidity              3
    AutoHotkey            3
    Hack                  2
    Pawn                  2
    CodeQL                2
    PowerShell            2
    Assembly              2
    Vim Script            2
    Vue                   2
    Elixir                2
    Gherkin               1
    QMake                 1
    CMake                 1
    Oz                    1
    Cuda                  1
    QML                   1
    ActionScript          1
    Roff                  1
    HCL                   1
    R                     1
    PureBasic             1
    Smarty                1
    Less                  1
    Svelte                1
    Haskell               1
    SourcePawn            1
    Name: count, dtype: int64 


​    

从输出结果可以看出两个结论：

·repositories项目名称存在多个重复项目

·language编程语言使用频率前三依次是JavaScript,Python,HTML

数值属性：数值属性，给出5数概括及缺失值的个数，包括stars_count标星数量, forks_count, issues_count, pull_requests, contributors贡献者人数


```python
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

numerical_attributes_list = ["stars_count", "forks_count", "issues_count", "pull_requests", "contributors"]

for attribute in numerical_attributes_list:
    print(attribute, ":")
    print(fiveNumberandnull(attribute, df), "\n")
```

    stars_count :
    缺省值null个数:0
    Minimum: 0
    Q1: 1.0
    Median: 12.0
    Q3: 65.25
    Maximum: 995
    None 
    
    forks_count :
    缺省值null个数:0
    Minimum: 0
    Q1: 1.0
    Median: 6.0
    Q3: 38.25
    Maximum: 973
    None 
    
    issues_count :
    缺省值null个数:0
    Minimum: 1
    Q1: 1.0
    Median: 2.0
    Q3: 6.0
    Maximum: 612
    None 
    
    pull_requests :
    缺省值null个数:0
    Minimum: 0
    Q1: 0.0
    Median: 0.0
    Q3: 2.0
    Maximum: 567
    None 
    
    contributors :
    缺省值null个数:0
    Minimum: 0
    Q1: 0.0
    Median: 2.0
    Q3: 4.0
    Maximum: 658
    None 


​    

从上面输出结果可以看出，stars_count标星数量、forks_count、issues_count、pull_requests、contributors贡献者人数都没有缺省值

#### 数据可视化

使用直方图、盒图等检查数据分布及离群点


```python
import matplotlib.pyplot as plt

numerical_attributes_list = ["stars_count", "forks_count", "issues_count", "pull_requests", "contributors"]
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
```

    stars_count 的直方图：




![png](/pic/output_9_1.png)
    


    stars_count 的盒图：




![png](/pic/output_9_3.png)
    


    forks_count 的直方图：




![png](/pic/output_9_5.png)
    


    forks_count 的盒图：




![png](/pic/output_9_7.png)
    


    issues_count 的直方图：




![png](/pic/output_9_9.png)
    


    issues_count 的盒图：




![png](/pic/output_9_11.png)
    


    pull_requests 的直方图：




![png](/pic/output_9_13.png)
    


    pull_requests 的盒图：




![png](/pic/output_9_15.png)
    


    contributors 的直方图：




![png](/pic/output_9_17.png)
    


    contributors 的盒图：




![png](/pic/output_9_19.png)
    


从上面输出可以看出：

·stars_count标星数量大部分位于0-100，少量项目stars高于200

·forks_count大部分位于0-100

·issues_count大部分位于0-50

·pull_requests大部分位于0-50

·contributors贡献者人数大部分位于0-50

#### 数据缺失的处理

经过统计，在这个数据集中，只有编程语言language存在缺失数据的现象


```python
# 统计缺失值数量
data = df['language']
null_num = data.isnull().sum()
print("编程语言language中的缺失值数量： ", null_num, "\n")

# 统计频数
counts = data.value_counts()
print(counts)
```

    编程语言language中的缺失值数量：  145 
    
    language
    JavaScript          253
    Python              155
    HTML                 72
    Java                 44
    CSS                  37
    TypeScript           37
    Dart                 36
    C++                  29
    Jupyter Notebook     29
    Ruby                 28
    C                    26
    Shell                25
    PHP                  16
    Go                   15
    Rust                 10
    Swift                10
    C#                    8
    Objective-C           8
    Kotlin                7
    Makefile              6
    Jinja                 5
    SCSS                  4
    CoffeeScript          3
    Perl                  3
    Dockerfile            3
    Solidity              3
    AutoHotkey            3
    Hack                  2
    Pawn                  2
    CodeQL                2
    PowerShell            2
    Assembly              2
    Vim Script            2
    Vue                   2
    Elixir                2
    Gherkin               1
    QMake                 1
    CMake                 1
    Oz                    1
    Cuda                  1
    QML                   1
    ActionScript          1
    Roff                  1
    HCL                   1
    R                     1
    PureBasic             1
    Smarty                1
    Less                  1
    Svelte                1
    Haskell               1
    SourcePawn            1
    Name: count, dtype: int64


下面分别使用三种方式处理缺失值：

将缺失部分剔除

用最高频率值来填补缺失值

通过属性的相关关系来填补缺失值

※※※ 由于缺失值为标称型数据，因此无法通过数据对象之间的相似性来填补缺失值


```python
print("一、将缺失部分剔除")
data_dele = data.dropna()
print("剔除缺失值前的数据集⼤⼩: ",data.shape)
print("剔除缺失值后的数据集⼤⼩: ", data_dele.shape)
print("剔除缺失值后缺失值统计：", data_dele.isnull().sum())
print()

print("二、用最高频率值来填补缺失值")
data_max_upda = data.fillna("JavaScript")
print("替换缺失值前的数据集⼤⼩: ",data.shape)
print("替换缺失值后的数据集⼤⼩: ", data_max_upda.shape)
print("将缺失部分剔除后缺失值统计：", data_max_upda.isnull().sum(), "个")
counts = data.value_counts()
counts_max_upda = data_max_upda.value_counts()
print("替换缺失值前频数统计\n", counts, "\n")
print("替换缺失值后频数统计\n", counts_max_upda, "\n")

print("三、通过属性的相关关系来填补缺失值")
for col in numerical_attributes_list:
    print("相关属性：", col)
    mean = df[col].mean()
    data_rela_upda = data.fillna(mean)
    print("替换缺失值前的数据集⼤⼩: ",data.shape)
    print("替换缺失值后的数据集⼤⼩: ", data_rela_upda.shape)
    print("将缺失部分剔除后缺失值统计：", data_rela_upda.isnull().sum(), "个")
    counts = data.value_counts()
    counts_rela_upda = data_rela_upda.value_counts()
    print("替换缺失值前频数统计\n", counts, "\n")
    print("替换缺失值后频数统计\n", counts_rela_upda, "\n")
```

    一、将缺失部分剔除
    剔除缺失值前的数据集⼤⼩:  (1052,)
    剔除缺失值后的数据集⼤⼩:  (907,)
    剔除缺失值后缺失值统计： 0
    
    二、用最高频率值来填补缺失值
    替换缺失值前的数据集⼤⼩:  (1052,)
    替换缺失值后的数据集⼤⼩:  (1052,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     language
    JavaScript          253
    Python              155
    HTML                 72
    Java                 44
    CSS                  37
    TypeScript           37
    Dart                 36
    C++                  29
    Jupyter Notebook     29
    Ruby                 28
    C                    26
    Shell                25
    PHP                  16
    Go                   15
    Rust                 10
    Swift                10
    C#                    8
    Objective-C           8
    Kotlin                7
    Makefile              6
    Jinja                 5
    SCSS                  4
    CoffeeScript          3
    Perl                  3
    Dockerfile            3
    Solidity              3
    AutoHotkey            3
    Hack                  2
    Pawn                  2
    CodeQL                2
    PowerShell            2
    Assembly              2
    Vim Script            2
    Vue                   2
    Elixir                2
    Gherkin               1
    QMake                 1
    CMake                 1
    Oz                    1
    Cuda                  1
    QML                   1
    ActionScript          1
    Roff                  1
    HCL                   1
    R                     1
    PureBasic             1
    Smarty                1
    Less                  1
    Svelte                1
    Haskell               1
    SourcePawn            1
    Name: count, dtype: int64 
    
    替换缺失值后频数统计
     language
    JavaScript          398
    Python              155
    HTML                 72
    Java                 44
    CSS                  37
    TypeScript           37
    Dart                 36
    Jupyter Notebook     29
    C++                  29
    Ruby                 28
    C                    26
    Shell                25
    PHP                  16
    Go                   15
    Rust                 10
    Swift                10
    C#                    8
    Objective-C           8
    Kotlin                7
    Makefile              6
    Jinja                 5
    SCSS                  4
    CoffeeScript          3
    Perl                  3
    Dockerfile            3
    Solidity              3
    AutoHotkey            3
    Hack                  2
    Pawn                  2
    CodeQL                2
    PowerShell            2
    Assembly              2
    Vim Script            2
    Vue                   2
    Elixir                2
    Gherkin               1
    QMake                 1
    CMake                 1
    Oz                    1
    Cuda                  1
    QML                   1
    ActionScript          1
    Roff                  1
    HCL                   1
    R                     1
    PureBasic             1
    Smarty                1
    Less                  1
    Svelte                1
    Haskell               1
    SourcePawn            1
    Name: count, dtype: int64 
    
    三、通过属性的相关关系来填补缺失值
    相关属性： stars_count
    替换缺失值前的数据集⼤⼩:  (1052,)
    替换缺失值后的数据集⼤⼩:  (1052,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     language
    JavaScript          253
    Python              155
    HTML                 72
    Java                 44
    CSS                  37
    TypeScript           37
    Dart                 36
    C++                  29
    Jupyter Notebook     29
    Ruby                 28
    C                    26
    Shell                25
    PHP                  16
    Go                   15
    Rust                 10
    Swift                10
    C#                    8
    Objective-C           8
    Kotlin                7
    Makefile              6
    Jinja                 5
    SCSS                  4
    CoffeeScript          3
    Perl                  3
    Dockerfile            3
    Solidity              3
    AutoHotkey            3
    Hack                  2
    Pawn                  2
    CodeQL                2
    PowerShell            2
    Assembly              2
    Vim Script            2
    Vue                   2
    Elixir                2
    Gherkin               1
    QMake                 1
    CMake                 1
    Oz                    1
    Cuda                  1
    QML                   1
    ActionScript          1
    Roff                  1
    HCL                   1
    R                     1
    PureBasic             1
    Smarty                1
    Less                  1
    Svelte                1
    Haskell               1
    SourcePawn            1
    Name: count, dtype: int64 
    
    替换缺失值后频数统计
     language
    JavaScript           253
    Python               155
    81.97623574144487    145
    HTML                  72
    Java                  44
    CSS                   37
    TypeScript            37
    Dart                  36
    Jupyter Notebook      29
    C++                   29
    Ruby                  28
    C                     26
    Shell                 25
    PHP                   16
    Go                    15
    Swift                 10
    Rust                  10
    C#                     8
    Objective-C            8
    Kotlin                 7
    Makefile               6
    Jinja                  5
    SCSS                   4
    AutoHotkey             3
    Dockerfile             3
    CoffeeScript           3
    Perl                   3
    Solidity               3
    Vim Script             2
    Pawn                   2
    Assembly               2
    PowerShell             2
    Hack                   2
    CodeQL                 2
    Vue                    2
    Elixir                 2
    Gherkin                1
    QMake                  1
    CMake                  1
    Oz                     1
    Cuda                   1
    QML                    1
    ActionScript           1
    Roff                   1
    HCL                    1
    R                      1
    PureBasic              1
    Smarty                 1
    Less                   1
    Svelte                 1
    Haskell                1
    SourcePawn             1
    Name: count, dtype: int64 
    
    相关属性： forks_count
    替换缺失值前的数据集⼤⼩:  (1052,)
    替换缺失值后的数据集⼤⼩:  (1052,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     language
    JavaScript          253
    Python              155
    HTML                 72
    Java                 44
    CSS                  37
    TypeScript           37
    Dart                 36
    C++                  29
    Jupyter Notebook     29
    Ruby                 28
    C                    26
    Shell                25
    PHP                  16
    Go                   15
    Rust                 10
    Swift                10
    C#                    8
    Objective-C           8
    Kotlin                7
    Makefile              6
    Jinja                 5
    SCSS                  4
    CoffeeScript          3
    Perl                  3
    Dockerfile            3
    Solidity              3
    AutoHotkey            3
    Hack                  2
    Pawn                  2
    CodeQL                2
    PowerShell            2
    Assembly              2
    Vim Script            2
    Vue                   2
    Elixir                2
    Gherkin               1
    QMake                 1
    CMake                 1
    Oz                    1
    Cuda                  1
    QML                   1
    ActionScript          1
    Roff                  1
    HCL                   1
    R                     1
    PureBasic             1
    Smarty                1
    Less                  1
    Svelte                1
    Haskell               1
    SourcePawn            1
    Name: count, dtype: int64 
    
    替换缺失值后频数统计
     language
    JavaScript            253
    Python                155
    53.884980988593156    145
    HTML                   72
    Java                   44
    CSS                    37
    TypeScript             37
    Dart                   36
    Jupyter Notebook       29
    C++                    29
    Ruby                   28
    C                      26
    Shell                  25
    PHP                    16
    Go                     15
    Swift                  10
    Rust                   10
    C#                      8
    Objective-C             8
    Kotlin                  7
    Makefile                6
    Jinja                   5
    SCSS                    4
    AutoHotkey              3
    Dockerfile              3
    CoffeeScript            3
    Perl                    3
    Solidity                3
    Vim Script              2
    Pawn                    2
    Assembly                2
    PowerShell              2
    Hack                    2
    CodeQL                  2
    Vue                     2
    Elixir                  2
    Gherkin                 1
    QMake                   1
    CMake                   1
    Oz                      1
    Cuda                    1
    QML                     1
    ActionScript            1
    Roff                    1
    HCL                     1
    R                       1
    PureBasic               1
    Smarty                  1
    Less                    1
    Svelte                  1
    Haskell                 1
    SourcePawn              1
    Name: count, dtype: int64 
    
    相关属性： issues_count
    替换缺失值前的数据集⼤⼩:  (1052,)
    替换缺失值后的数据集⼤⼩:  (1052,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     language
    JavaScript          253
    Python              155
    HTML                 72
    Java                 44
    CSS                  37
    TypeScript           37
    Dart                 36
    C++                  29
    Jupyter Notebook     29
    Ruby                 28
    C                    26
    Shell                25
    PHP                  16
    Go                   15
    Rust                 10
    Swift                10
    C#                    8
    Objective-C           8
    Kotlin                7
    Makefile              6
    Jinja                 5
    SCSS                  4
    CoffeeScript          3
    Perl                  3
    Dockerfile            3
    Solidity              3
    AutoHotkey            3
    Hack                  2
    Pawn                  2
    CodeQL                2
    PowerShell            2
    Assembly              2
    Vim Script            2
    Vue                   2
    Elixir                2
    Gherkin               1
    QMake                 1
    CMake                 1
    Oz                    1
    Cuda                  1
    QML                   1
    ActionScript          1
    Roff                  1
    HCL                   1
    R                     1
    PureBasic             1
    Smarty                1
    Less                  1
    Svelte                1
    Haskell               1
    SourcePawn            1
    Name: count, dtype: int64 
    
    替换缺失值后频数统计
     language
    JavaScript           253
    Python               155
    8.656844106463879    145
    HTML                  72
    Java                  44
    CSS                   37
    TypeScript            37
    Dart                  36
    Jupyter Notebook      29
    C++                   29
    Ruby                  28
    C                     26
    Shell                 25
    PHP                   16
    Go                    15
    Swift                 10
    Rust                  10
    C#                     8
    Objective-C            8
    Kotlin                 7
    Makefile               6
    Jinja                  5
    SCSS                   4
    AutoHotkey             3
    Dockerfile             3
    CoffeeScript           3
    Perl                   3
    Solidity               3
    Vim Script             2
    Pawn                   2
    Assembly               2
    PowerShell             2
    Hack                   2
    CodeQL                 2
    Vue                    2
    Elixir                 2
    Gherkin                1
    QMake                  1
    CMake                  1
    Oz                     1
    Cuda                   1
    QML                    1
    ActionScript           1
    Roff                   1
    HCL                    1
    R                      1
    PureBasic              1
    Smarty                 1
    Less                   1
    Svelte                 1
    Haskell                1
    SourcePawn             1
    Name: count, dtype: int64 
    
    相关属性： pull_requests
    替换缺失值前的数据集⼤⼩:  (1052,)
    替换缺失值后的数据集⼤⼩:  (1052,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     language
    JavaScript          253
    Python              155
    HTML                 72
    Java                 44
    CSS                  37
    TypeScript           37
    Dart                 36
    C++                  29
    Jupyter Notebook     29
    Ruby                 28
    C                    26
    Shell                25
    PHP                  16
    Go                   15
    Rust                 10
    Swift                10
    C#                    8
    Objective-C           8
    Kotlin                7
    Makefile              6
    Jinja                 5
    SCSS                  4
    CoffeeScript          3
    Perl                  3
    Dockerfile            3
    Solidity              3
    AutoHotkey            3
    Hack                  2
    Pawn                  2
    CodeQL                2
    PowerShell            2
    Assembly              2
    Vim Script            2
    Vue                   2
    Elixir                2
    Gherkin               1
    QMake                 1
    CMake                 1
    Oz                    1
    Cuda                  1
    QML                   1
    ActionScript          1
    Roff                  1
    HCL                   1
    R                     1
    PureBasic             1
    Smarty                1
    Less                  1
    Svelte                1
    Haskell               1
    SourcePawn            1
    Name: count, dtype: int64 
    
    替换缺失值后频数统计
     language
    JavaScript            253
    Python                155
    4.3745247148288975    145
    HTML                   72
    Java                   44
    CSS                    37
    TypeScript             37
    Dart                   36
    Jupyter Notebook       29
    C++                    29
    Ruby                   28
    C                      26
    Shell                  25
    PHP                    16
    Go                     15
    Swift                  10
    Rust                   10
    C#                      8
    Objective-C             8
    Kotlin                  7
    Makefile                6
    Jinja                   5
    SCSS                    4
    AutoHotkey              3
    Dockerfile              3
    CoffeeScript            3
    Perl                    3
    Solidity                3
    Vim Script              2
    Pawn                    2
    Assembly                2
    PowerShell              2
    Hack                    2
    CodeQL                  2
    Vue                     2
    Elixir                  2
    Gherkin                 1
    QMake                   1
    CMake                   1
    Oz                      1
    Cuda                    1
    QML                     1
    ActionScript            1
    Roff                    1
    HCL                     1
    R                       1
    PureBasic               1
    Smarty                  1
    Less                    1
    Svelte                  1
    Haskell                 1
    SourcePawn              1
    Name: count, dtype: int64 
    
    相关属性： contributors
    替换缺失值前的数据集⼤⼩:  (1052,)
    替换缺失值后的数据集⼤⼩:  (1052,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     language
    JavaScript          253
    Python              155
    HTML                 72
    Java                 44
    CSS                  37
    TypeScript           37
    Dart                 36
    C++                  29
    Jupyter Notebook     29
    Ruby                 28
    C                    26
    Shell                25
    PHP                  16
    Go                   15
    Rust                 10
    Swift                10
    C#                    8
    Objective-C           8
    Kotlin                7
    Makefile              6
    Jinja                 5
    SCSS                  4
    CoffeeScript          3
    Perl                  3
    Dockerfile            3
    Solidity              3
    AutoHotkey            3
    Hack                  2
    Pawn                  2
    CodeQL                2
    PowerShell            2
    Assembly              2
    Vim Script            2
    Vue                   2
    Elixir                2
    Gherkin               1
    QMake                 1
    CMake                 1
    Oz                    1
    Cuda                  1
    QML                   1
    ActionScript          1
    Roff                  1
    HCL                   1
    R                     1
    PureBasic             1
    Smarty                1
    Less                  1
    Svelte                1
    Haskell               1
    SourcePawn            1
    Name: count, dtype: int64 
    
    替换缺失值后频数统计
     language
    JavaScript           253
    Python               155
    8.364068441064639    145
    HTML                  72
    Java                  44
    CSS                   37
    TypeScript            37
    Dart                  36
    Jupyter Notebook      29
    C++                   29
    Ruby                  28
    C                     26
    Shell                 25
    PHP                   16
    Go                    15
    Swift                 10
    Rust                  10
    C#                     8
    Objective-C            8
    Kotlin                 7
    Makefile               6
    Jinja                  5
    SCSS                   4
    AutoHotkey             3
    Dockerfile             3
    CoffeeScript           3
    Perl                   3
    Solidity               3
    Vim Script             2
    Pawn                   2
    Assembly               2
    PowerShell             2
    Hack                   2
    CodeQL                 2
    Vue                    2
    Elixir                 2
    Gherkin                1
    QMake                  1
    CMake                  1
    Oz                     1
    Cuda                   1
    QML                    1
    ActionScript           1
    Roff                   1
    HCL                    1
    R                      1
    PureBasic              1
    Smarty                 1
    Less                   1
    Svelte                 1
    Haskell                1
    SourcePawn             1
    Name: count, dtype: int64 


​    

### 二、对Alzheimer Disease and Healthy Aging Data In US的数据处理

#### 数据摘要

标称属性： 标称属性，给出每个可能取值的频数，包括YearStart、YearEnd、LocationAbbr、LocationDesc、Datasource、Class、Topic、Question、Data_Value_Unit、DataValueTypeID、Data_Value_Type、StratificationCategory1、Stratification1、StratificationCategory2、Stratification2、Geolocation、ClassID、TopicID、QuestionID、LocationID、StratificationCategoryID1、StratificationID1、StratificationCategoryID2、StratificationID2


```python
import numpy as np
import pandas as pd
from pandas import DataFrame

def loader(filepath):   #读取数据
    df = pd.read_csv(filepath, header=0)
    return df

def count(str,data): #输出标称数据频数
    return data[str].value_counts()

df = loader("./Alzheimer Disease and Healthy Aging Data In US/Alzheimer Disease and Healthy Aging Data In US.csv")
df.replace('.', np.nan, inplace=True)
nominal_attribute_list = ["YearStart", "YearEnd", "LocationAbbr", "LocationDesc", "Datasource", "Class", "Topic", "Question", "Data_Value_Unit", "DataValueTypeID", "Data_Value_Type", "StratificationCategory1", "Stratification1", "StratificationCategory2", "Stratification2", "Geolocation", "ClassID", "TopicID", "QuestionID", "LocationID", "StratificationCategoryID1", "StratificationID1", "StratificationCategoryID2", "StratificationID2"]

for attribute in nominal_attribute_list:
    print(attribute, "的频数：")
    print(count(attribute, df), "\n")
```

    C:\Users\LiuSiyuan\AppData\Local\Temp\ipykernel_17588\3945464940.py:6: DtypeWarning: Columns (13,14) have mixed types. Specify dtype option on import or set low_memory=False.
      df = pd.read_csv(filepath, header=0)


    YearStart 的频数：
    YearStart
    2015    45980
    2020    36006
    2019    34354
    2016    34145
    2017    33429
    2018    30548
    Name: count, dtype: int64 
    
    YearEnd 的频数：
    YearEnd
    2020    46966
    2015    35020
    2019    34354
    2016    34145
    2017    33429
    2018    30548
    Name: count, dtype: int64 
    
    LocationAbbr 的频数：
    LocationAbbr
    US      4644
    WEST    4638
    NRE     4614
    MDW     4611
    OR      4565
    NY      4557
    SOU     4542
    UT      4222
    OH      3955
    GA      3951
    MD      3919
    HI      3907
    TN      3879
    MI      3796
    VA      3758
    FL      3753
    ME      3733
    TX      3699
    NV      3696
    DC      3684
    WV      3682
    MS      3677
    PA      3648
    NM      3635
    AL      3633
    KY      3623
    AK      3611
    SC      3592
    NJ      3589
    AZ      3582
    MO      3573
    IL      3571
    IN      3570
    WI      3569
    LA      3563
    MN      3555
    NE      3546
    CT      3543
    RI      3534
    OK      3526
    SD      3526
    ND      3514
    KS      3510
    ID      3507
    IA      3501
    AR      3498
    WY      3494
    CA      3447
    CO      3390
    NC      3349
    WA      3348
    MT      3348
    DE      3346
    NH      3284
    VT      3278
    MA      3174
    PR      2797
    GU      2703
    VI       503
    Name: count, dtype: int64 
    
    LocationDesc 的频数：
    LocationDesc
    United States, DC & Territories    4644
    West                               4638
    Northeast                          4614
    Midwest                            4611
    Oregon                             4565
    New York                           4557
    South                              4542
    Utah                               4222
    Ohio                               3955
    Georgia                            3951
    Maryland                           3919
    Hawaii                             3907
    Tennessee                          3879
    Michigan                           3796
    Virginia                           3758
    Florida                            3753
    Maine                              3733
    Texas                              3699
    Nevada                             3696
    District of Columbia               3684
    West Virginia                      3682
    Mississippi                        3677
    Pennsylvania                       3648
    New Mexico                         3635
    Alabama                            3633
    Kentucky                           3623
    Alaska                             3611
    South Carolina                     3592
    New Jersey                         3589
    Arizona                            3582
    Missouri                           3573
    Illinois                           3571
    Indiana                            3570
    Wisconsin                          3569
    Louisiana                          3563
    Minnesota                          3555
    Nebraska                           3546
    Connecticut                        3543
    Rhode Island                       3534
    Oklahoma                           3526
    South Dakota                       3526
    North Dakota                       3514
    Kansas                             3510
    Idaho                              3507
    Iowa                               3501
    Arkansas                           3498
    Wyoming                            3494
    California                         3447
    Colorado                           3390
    North Carolina                     3349
    Washington                         3348
    Montana                            3348
    Delaware                           3346
    New Hampshire                      3284
    Vermont                            3278
    Massachusetts                      3174
    Puerto Rico                        2797
    Guam                               2703
    Virgin Islands                      503
    Name: count, dtype: int64 
    
    Datasource 的频数：
    Datasource
    BRFSS    214462
    Name: count, dtype: int64 
    
    Class 的频数：
    Class
    Overall Health                         71694
    Screenings and Vaccines                46867
    Nutrition/Physical Activity/Obesity    24851
    Cognitive Decline                      19180
    Caregiving                             18671
    Mental Health                          16600
    Smoking and Alcohol Use                16599
    Name: count, dtype: int64 
    
    Topic 的频数：
    Topic
    Obesity                                                                                                   8300
    Influenza vaccine within past year                                                                        8300
    Physically unhealthy days (mean number of days)                                                           8300
    Frequent mental distress                                                                                  8300
    Current smoking                                                                                           8300
    Lifetime diagnosis of depression                                                                          8300
    No leisure-time physical activity within past month                                                       8300
    Self-rated health (fair to poor health)                                                                   8299
    Self-rated health (good to excellent health)                                                              8299
    Binge drinking within past 30 days                                                                        8299
    Ever had pneumococcal vaccine                                                                             8268
    Recent activity limitations in past month                                                                 8233
    Disability status, including sensory or mobility limitations                                              6917
    Arthritis among older adults                                                                              5511
    Fair or poor health among older adults with arthritis                                                     5447
    Subjective cognitive decline or memory loss among older adults                                            5088
    Diabetes screening within past 3 years                                                                    4808
    Talked with health care professional about subjective cognitive decline or memory loss                    4700
    Need assistance with day-to-day activities because of subjective cognitive decline or memory loss         4696
    Functional difficulties associated with subjective cognitive decline or memory loss among older adults    4696
    Fall with injury within last year                                                                         4173
    Colorectal cancer screening                                                                               4173
    Oral health:  tooth retention                                                                             4172
    Prevalence of sufficient sleep                                                                            4171
    Eating 3 or more vegetables daily                                                                         4127
    High blood pressure ever                                                                                  4127
    Cholesterol checked in past 5 years                                                                       4127
    Eating 2 or more fruits daily                                                                             4124
    Taking medication for high blood pressure                                                                 4108
    Severe joint pain among older adults with arthritis                                                       4064
    Provide care for a friend or family member in past month                                                  3848
    Expect to provide care for someone in the next two years                                                  3797
    Provide care for someone with cognitive impairment within the past month                                  3682
    Duration of caregiving among older adults                                                                 3681
    Intensity of caregiving among older adults                                                                3663
    Up-to-date with recommended vaccines and screenings - Women                                               3280
    Up-to-date with recommended vaccines and screenings - Men                                                 3271
    Mammogram within past 2 years                                                                             3271
    Pap test within past 3 years                                                                              3242
    Name: count, dtype: int64 
    
    Question 的频数：
    Question
    Percentage of older adults who are currently obese, with a body mass index (BMI) of 30 or more                                                                               8300
    Percentage of older adults who reported influenza vaccine within the past year                                                                                               8300
    Physically unhealthy days (mean number of days in past month)                                                                                                                8300
    Percentage of older adults who are experiencing frequent mental distress                                                                                                     8300
    Percentage of older adults who have smoked at least 100 cigarettes in their entire life and still smoke every day or some days                                               8300
    Percentage of older adults with a lifetime diagnosis of depression                                                                                                           8300
    Percentage of older adults who have not had any leisure time physical activity in the past month                                                                             8300
    Percentage of older adults who self-reported that their health is "fair" or "poor"                                                                                           8299
    Percentage of older adults who self-reported that their health is "good", "very good", or "excellent"                                                                        8299
    Percentage of older adults who reported binge drinking within the past 30 days                                                                                               8299
    Percentage of at risk adults (have diabetes, asthma, cardiovascular disease or currently smoke) who ever had a pneumococcal vaccine                                          8268
    Mean number of days with activity limitations in the past month                                                                                                              8233
    Percentage of older adults who report having a disability (includes limitations related to sensory or mobility impairments or a physical, mental, or emotional condition)    6917
    Percentage of older adults ever told they have arthritis                                                                                                                     5511
    Fair or poor health among older adults with doctor-diagnosed arthritis                                                                                                       5447
    Percentage of older adults who reported subjective cognitive decline or memory loss that is happening more often or is getting worse in the preceding 12 months              5088
    Percentage of older adults without diabetes who reported a blood sugar or diabetes test within 3 years                                                                       4808
    Percentage of older adults with subjective cognitive decline or memory loss who reported talking with a health care professional about it                                    4700
    Percentage of older adults who reported that as a result of subjective cognitive decline or memory loss that they need assistance with day-to-day activities                 4696
    Percentage of older adults who reported subjective cognitive decline or memory loss that interferes with their ability to engage in social activities or household chores    4696
    Percentage of older adults who have fallen and sustained an injury within last year                                                                                          4173
    Percentage of older adults who had either a home blood stool test within the past year or a sigmoidoscopy or colonoscopy within the past 10 years                            4173
    Percentage of older adults who report having lost 5 or fewer teeth due to decay or gum disease                                                                               4172
    Percentage of older adults getting sufficient sleep (>6 hours)                                                                                                               4171
    Percentage of older adults who are eating 3 or more vegetables daily                                                                                                         4127
    Percentage of older adults who have ever been told by a health professional that they have high blood pressure                                                               4127
    Percentage of older adults who had a cholesterol screening within the past 5 years                                                                                           4127
    Percentage of older adults who are eating 2 or more fruits daily                                                                                                             4124
    Percentage of older adults who have been told they have high blood pressure who report currently taking medication for their high blood pressure                             4108
    Severe joint pain due to arthritis among older adults with doctor-diagnosed arthritis                                                                                        4064
    Percentage of older adults who provided care for a friend or family member within the past month                                                                             3848
    Percentage of older adults currently not providing care who expect to provide care for someone with health problems in the next two years                                    3797
    Percentage of older adults who provided care for someone with dementia or other cognitive impairment within the past month                                                   3682
    Percentage of older adults who provided care to a friend or family member for six months or more                                                                             3681
    Average of 20 or more hours of care per week provided to a friend or family member                                                                                           3663
    Percentage of older adult women who are up to date with select clinical preventive services                                                                                  3280
    Percentage of older adult men who are up to date with select clinical preventive services                                                                                    3271
    Percentage of older adult women who have received a mammogram within the past 2 years                                                                                        3271
    Percentage of older adult women with an intact cervix who had a Pap test within the past 3 years                                                                             3242
    Name: count, dtype: int64 
    
    Data_Value_Unit 的频数：
    Data_Value_Unit
    %         197929
    Number     16533
    Name: count, dtype: int64 
    
    DataValueTypeID 的频数：
    DataValueTypeID
    PRCTG    197929
    MEAN      16533
    Name: count, dtype: int64 
    
    Data_Value_Type 的频数：
    Data_Value_Type
    Percentage    197929
    Mean           16533
    Name: count, dtype: int64 
    
    StratificationCategory1 的频数：
    StratificationCategory1
    Age Group    214462
    Name: count, dtype: int64 
    
    Stratification1 的频数：
    Stratification1
    Overall              71919
    50-64 years          71528
    65 years or older    71015
    Name: count, dtype: int64 
    
    StratificationCategory2 的频数：
    StratificationCategory2
    Race/Ethnicity    134959
    Gender             51834
    Name: count, dtype: int64 
    
    Stratification2 的频数：
    Stratification2
    White, non-Hispanic         27633
    Hispanic                    27525
    Black, non-Hispanic         26968
    Native Am/Alaskan Native    26571
    Asian/Pacific Islander      26262
    Female                      26091
    Male                        25743
    Name: count, dtype: int64 
    
    Geolocation 的频数：
    Geolocation
    POINT (-120.1550313 44.56744942)    4565
    POINT (-75.54397043 42.82700103)    4557
    POINT (-111.5871306 39.36070017)    4222
    POINT (-82.40426006 40.06021014)    3955
    POINT (-83.62758035 32.83968109)    3951
    POINT (-76.60926011 39.29058096)    3919
    POINT (-157.8577494 21.30485044)    3907
    POINT (-85.77449091 35.68094058)    3879
    POINT (-84.71439027 44.66131954)    3796
    POINT (-78.45789046 37.54268067)    3758
    POINT (-81.92896054 28.93204038)    3753
    POINT (-68.98503134 45.25422889)    3733
    POINT (-99.42677021 31.82724041)    3699
    POINT (-117.0718406 39.49324039)    3696
    POINT (-77.036871 38.907192)        3684
    POINT (-80.71264013 38.6655102)     3682
    POINT (-89.53803082 32.7455101)     3677
    POINT (-77.86070029 40.79373015)    3648
    POINT (-106.240581 34.52088095)     3635
    POINT (-86.63186076 32.84057112)    3633
    POINT (-84.77497105 37.64597027)    3623
    POINT (-147.722059 64.84507996)     3611
    POINT (-81.04537121 33.9988213)     3592
    POINT (-74.27369129 40.13057005)    3589
    POINT (-111.7638113 34.86597028)    3582
    POINT (-92.56630005 38.63579078)    3573
    POINT (-88.99771018 40.48501028)    3571
    POINT (-86.14996019 39.76691045)    3570
    POINT (-89.81637074 44.39319117)    3569
    POINT (-92.44568007 31.31266064)    3563
    POINT (-94.7942005 46.35564874)     3555
    POINT (-99.36572062 41.64104099)    3546
    POINT (-72.64984095 41.56266102)    3543
    POINT (-71.52247031 41.70828019)    3534
    POINT (-97.52107021 35.47203136)    3526
    POINT (-100.3735306 44.35313005)    3526
    POINT (-100.118421 47.47531978)     3514
    POINT (-98.20078123 38.3477403)     3510
    POINT (-114.36373 43.68263001)      3507
    POINT (-93.81649056 42.46940091)    3501
    POINT (-92.27449074 34.74865012)    3498
    POINT (-108.1098304 43.23554134)    3494
    POINT (-120.9999995 37.63864012)    3447
    POINT (-106.1336109 38.84384076)    3390
    POINT (-79.15925046 35.46622098)    3349
    POINT (-109.4244206 47.06652897)    3348
    POINT (-120.4700108 47.52227863)    3348
    POINT (-75.57774117 39.00883067)    3346
    POINT (-71.50036092 43.65595011)    3284
    POINT (-72.51764079 43.62538124)    3278
    POINT (-72.08269067 42.27687047)    3174
    POINT (-66.590149 18.220833)        2797
    POINT (144.793731 13.444304)        2703
    POINT (-64.896335 18.335765)         503
    Name: count, dtype: int64 
    
    ClassID 的频数：
    ClassID
    C01    71694
    C03    46867
    C02    24851
    C06    19180
    C07    18671
    C05    16600
    C04    16599
    Name: count, dtype: int64 
    
    TopicID 的频数：
    TopicID
    TNC04    8300
    TSC08    8300
    TOC01    8300
    TMC01    8300
    TAC01    8300
    TMC03    8300
    TNC03    8300
    TOC07    8299
    TOC08    8299
    TAC03    8299
    TSC09    8268
    TOC03    8233
    TOC10    6917
    TOC11    5511
    TOC13    5447
    TCC01    5088
    TSC04    4808
    TCC04    4700
    TCC03    4696
    TCC02    4696
    TOC06    4173
    TSC02    4173
    TOC05    4172
    TOC09    4171
    TNC02    4127
    TSC07    4127
    TSC06    4127
    TNC01    4124
    TOC04    4108
    TOC12    4064
    TGC01    3848
    TGC02    3797
    TGC05    3682
    TGC03    3681
    TGC04    3663
    TSC11    3280
    TSC10    3271
    TSC01    3271
    TSC03    3242
    Name: count, dtype: int64 
    
    QuestionID 的频数：
    QuestionID
    Q13    8300
    Q18    8300
    Q08    8300
    Q03    8300
    Q17    8300
    Q27    8300
    Q16    8300
    Q32    8299
    Q33    8299
    Q21    8299
    Q09    8268
    Q35    8233
    Q46    6917
    Q43    5511
    Q45    5447
    Q30    5088
    Q19    4808
    Q42    4700
    Q41    4696
    Q31    4696
    Q05    4173
    Q15    4173
    Q07    4172
    Q34    4171
    Q02    4127
    Q22    4127
    Q14    4127
    Q01    4124
    Q04    4108
    Q44    4064
    Q36    3848
    Q37    3797
    Q40    3682
    Q38    3681
    Q39    3663
    Q11    3280
    Q10    3271
    Q12    3271
    Q20    3242
    Name: count, dtype: int64 
    
    LocationID 的频数：
    LocationID
    59      4644
    9004    4638
    9001    4614
    9002    4611
    41      4565
    36      4557
    9003    4542
    49      4222
    39      3955
    13      3951
    24      3919
    15      3907
    47      3879
    26      3796
    51      3758
    12      3753
    23      3733
    48      3699
    32      3696
    11      3684
    54      3682
    28      3677
    42      3648
    35      3635
    1       3633
    21      3623
    2       3611
    45      3592
    34      3589
    4       3582
    29      3573
    17      3571
    18      3570
    55      3569
    22      3563
    27      3555
    31      3546
    9       3543
    44      3534
    40      3526
    46      3526
    38      3514
    20      3510
    16      3507
    19      3501
    5       3498
    56      3494
    6       3447
    8       3390
    37      3349
    53      3348
    30      3348
    10      3346
    33      3284
    50      3278
    25      3174
    72      2797
    66      2703
    78       503
    Name: count, dtype: int64 
    
    StratificationCategoryID1 的频数：
    StratificationCategoryID1
    AGE    214462
    Name: count, dtype: int64 
    
    StratificationID1 的频数：
    StratificationID1
    AGE_OVERALL    71919
    5064           71528
    65PLUS         71015
    Name: count, dtype: int64 
    
    StratificationCategoryID2 的频数：
    StratificationCategoryID2
    RACE       134959
    GENDER      51834
    OVERALL     27669
    Name: count, dtype: int64 
    
    StratificationID2 的频数：
    StratificationID2
    OVERALL    27669
    WHT        27633
    HIS        27525
    BLK        26968
    NAA        26571
    ASN        26262
    FEMALE     26091
    MALE       25743
    Name: count, dtype: int64 


​    

数值属性：数值属性，给出5数概括及缺失值的个数，包括Data_Value、Data_Value_Alt、Low_Confidence_Limit、High_Confidence_Limit


```python
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

numerical_attributes_list = ["Data_Value", "Data_Value_Alt", "Low_Confidence_Limit", "High_Confidence_Limit"]
for attribute in numerical_attributes_list:
    df[attribute] = df[attribute].astype('float64')
    print(attribute, ":")
    print(fiveNumberandnull(attribute, df), "\n")
```

    Data_Value :
    缺省值null个数:69833
    Minimum: 0.0
    Q1: 15.3
    Median: 32.5
    Q3: 56.8
    Maximum: 100.0
    None 
    
    Data_Value_Alt :
    缺省值null个数:69833
    Minimum: 0.0
    Q1: 15.3
    Median: 32.5
    Q3: 56.8
    Maximum: 100.0
    None 
    
    Low_Confidence_Limit :
    缺省值null个数:70009
    Minimum: 0.0
    Q1: 12.0
    Median: 26.9
    Q3: 49.1
    Maximum: 99.6
    None 
    
    High_Confidence_Limit :
    缺省值null个数:70009
    Minimum: 1.4
    Q1: 19.0
    Median: 38.5
    Q3: 64.7
    Maximum: 100.0
    None 


​    

从上面输出结果可以看出，Data_Value、Data_Value_Alt、Low_Confidence_Limit、High_Confidence_Limit均有较多缺失值

#### 数据可视化

使用直方图、盒图等检查数据分布及离群点


```python
import matplotlib.pyplot as plt

numerical_attributes_list = ["Data_Value", "Data_Value_Alt", "Low_Confidence_Limit", "High_Confidence_Limit"]
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
```

    Data_Value 的直方图：




![png](/pic/output_21_1.png)
    


    Data_Value 的盒图：




![png](/pic/output_21_3.png)
    


    Data_Value_Alt 的直方图：




![png](/pic/output_21_5.png)
    


    Data_Value_Alt 的盒图：




![png](/pic/output_21_7.png)
    


    Low_Confidence_Limit 的直方图：




![png](/pic/output_21_9.png)
    


    Low_Confidence_Limit 的盒图：




![png](/pic/output_21_11.png)
    


    High_Confidence_Limit 的直方图：




![png](/pic/output_21_13.png)
    


    High_Confidence_Limit 的盒图：




![png](/pic/output_21_15.png)
    


#### 数据缺失的处理

经过统计，在这个数据集中，Data_Value、Data_Value_Alt、Low_Confidence_Limit、High_Confidence_Limit均有较多缺失值


```python
# 统计缺失值数量
for attribute in numerical_attributes_list:
    null_num = df[attribute].isnull().sum()
    print("编程语言language中的缺失值数量： ", null_num, "\n")
```

    编程语言language中的缺失值数量：  69833 
    
    编程语言language中的缺失值数量：  69833 
    
    编程语言language中的缺失值数量：  70009 
    
    编程语言language中的缺失值数量：  70009 


​    

下面分别使用三种方式处理缺失值：

将缺失部分剔除

通过属性的相关关系来填补缺失值

通过数据对象之间的相似性来填补缺失值

由于缺失值为数值型数据，因此无法用最高频率值来填补缺失值


```python
from sklearn.impute import KNNImputer

for attribute in numerical_attributes_list:
    data = df[attribute]
    print("一、将缺失部分剔除")
    data_dele = data.dropna()
    print("剔除缺失值前的数据集⼤⼩: ",data.shape)
    print("剔除缺失值后的数据集⼤⼩: ", data_dele.shape)
    print("剔除缺失值后缺失值统计：", data_dele.isnull().sum())
    print()

    print("二、通过属性的相关关系来填补缺失值")
    for col in numerical_attributes_list:
        print("相关属性：", col)
        mean = df[col].mean()
        data_rela_upda = data.fillna(mean)
        print("替换缺失值前的数据集⼤⼩: ",data.shape)
        print("替换缺失值后的数据集⼤⼩: ", data_rela_upda.shape)
        print("将缺失部分剔除后缺失值统计：", data_rela_upda.isnull().sum(), "个")
        counts = data.value_counts()
        counts_rela_upda = data_rela_upda.value_counts()
        print("替换缺失值前频数统计\n", counts, "\n")
        print("替换缺失值后频数统计\n", counts_rela_upda, "\n")

print("三、通过数据对象之间的相似性来填补缺失值")
df_num = df[numerical_attributes_list].astype('float64')
knn_imputer = KNNImputer(n_neighbors=2)
filled_values = knn_imputer.fit_transform(df_num)
df_filled = pd.DataFrame(filled_values, columns=df.columns)  

    # 打印填补后的DataFrame  
print(df_filled)

```

    一、将缺失部分剔除
    剔除缺失值前的数据集⼤⼩:  (214462,)
    剔除缺失值后的数据集⼤⼩:  (144629,)
    剔除缺失值后缺失值统计： 0
    
    二、通过属性的相关关系来填补缺失值
    相关属性： Data_Value
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     Data_Value
    6.0     473
    6.3     467
    6.1     459
    5.8     458
    5.2     456
           ... 
    99.7      2
    0.5       1
    0.6       1
    0.3       1
    0.2       1
    Name: count, Length: 999, dtype: int64 
    
    替换缺失值后频数统计
     Data_Value
    37.341956    69833
    6.000000       473
    6.300000       467
    6.100000       459
    5.800000       458
                 ...  
    99.700000        2
    0.500000         1
    0.600000         1
    0.300000         1
    0.200000         1
    Name: count, Length: 1000, dtype: int64 
    
    相关属性： Data_Value_Alt
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     Data_Value
    6.0     473
    6.3     467
    6.1     459
    5.8     458
    5.2     456
           ... 
    99.7      2
    0.5       1
    0.6       1
    0.3       1
    0.2       1
    Name: count, Length: 999, dtype: int64 
    
    替换缺失值后频数统计
     Data_Value
    37.341956    69833
    6.000000       473
    6.300000       467
    6.100000       459
    5.800000       458
                 ...  
    99.700000        2
    0.500000         1
    0.600000         1
    0.300000         1
    0.200000         1
    Name: count, Length: 1000, dtype: int64 
    
    相关属性： Low_Confidence_Limit
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     Data_Value
    6.0     473
    6.3     467
    6.1     459
    5.8     458
    5.2     456
           ... 
    99.7      2
    0.5       1
    0.6       1
    0.3       1
    0.2       1
    Name: count, Length: 999, dtype: int64 
    
    替换缺失值后频数统计
     Data_Value
    32.736785    69833
    6.000000       473
    6.300000       467
    6.100000       459
    5.800000       458
                 ...  
    99.700000        2
    0.500000         1
    0.600000         1
    0.300000         1
    0.200000         1
    Name: count, Length: 1000, dtype: int64 
    
    相关属性： High_Confidence_Limit
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     Data_Value
    6.0     473
    6.3     467
    6.1     459
    5.8     458
    5.2     456
           ... 
    99.7      2
    0.5       1
    0.6       1
    0.3       1
    0.2       1
    Name: count, Length: 999, dtype: int64 
    
    替换缺失值后频数统计
     Data_Value
    42.244436    69833
    6.000000       473
    6.300000       467
    6.100000       459
    5.800000       458
                 ...  
    99.700000        2
    0.500000         1
    0.600000         1
    0.300000         1
    0.200000         1
    Name: count, Length: 1000, dtype: int64 
    
    一、将缺失部分剔除
    剔除缺失值前的数据集⼤⼩:  (214462,)
    剔除缺失值后的数据集⼤⼩:  (144629,)
    剔除缺失值后缺失值统计： 0
    
    二、通过属性的相关关系来填补缺失值
    相关属性： Data_Value
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     Data_Value_Alt
    6.0     473
    6.3     467
    6.1     459
    5.8     458
    5.2     456
           ... 
    99.7      2
    0.5       1
    0.6       1
    0.3       1
    0.2       1
    Name: count, Length: 999, dtype: int64 
    
    替换缺失值后频数统计
     Data_Value_Alt
    37.341956    69833
    6.000000       473
    6.300000       467
    6.100000       459
    5.800000       458
                 ...  
    99.700000        2
    0.500000         1
    0.600000         1
    0.300000         1
    0.200000         1
    Name: count, Length: 1000, dtype: int64 
    
    相关属性： Data_Value_Alt
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     Data_Value_Alt
    6.0     473
    6.3     467
    6.1     459
    5.8     458
    5.2     456
           ... 
    99.7      2
    0.5       1
    0.6       1
    0.3       1
    0.2       1
    Name: count, Length: 999, dtype: int64 
    
    替换缺失值后频数统计
     Data_Value_Alt
    37.341956    69833
    6.000000       473
    6.300000       467
    6.100000       459
    5.800000       458
                 ...  
    99.700000        2
    0.500000         1
    0.600000         1
    0.300000         1
    0.200000         1
    Name: count, Length: 1000, dtype: int64 
    
    相关属性： Low_Confidence_Limit
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     Data_Value_Alt
    6.0     473
    6.3     467
    6.1     459
    5.8     458
    5.2     456
           ... 
    99.7      2
    0.5       1
    0.6       1
    0.3       1
    0.2       1
    Name: count, Length: 999, dtype: int64 
    
    替换缺失值后频数统计
     Data_Value_Alt
    32.736785    69833
    6.000000       473
    6.300000       467
    6.100000       459
    5.800000       458
                 ...  
    99.700000        2
    0.500000         1
    0.600000         1
    0.300000         1
    0.200000         1
    Name: count, Length: 1000, dtype: int64 
    
    相关属性： High_Confidence_Limit
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     Data_Value_Alt
    6.0     473
    6.3     467
    6.1     459
    5.8     458
    5.2     456
           ... 
    99.7      2
    0.5       1
    0.6       1
    0.3       1
    0.2       1
    Name: count, Length: 999, dtype: int64 
    
    替换缺失值后频数统计
     Data_Value_Alt
    42.244436    69833
    6.000000       473
    6.300000       467
    6.100000       459
    5.800000       458
                 ...  
    99.700000        2
    0.500000         1
    0.600000         1
    0.300000         1
    0.200000         1
    Name: count, Length: 1000, dtype: int64 
    
    一、将缺失部分剔除
    剔除缺失值前的数据集⼤⼩:  (214462,)
    剔除缺失值后的数据集⼤⼩:  (144453,)
    剔除缺失值后缺失值统计： 0
    
    二、通过属性的相关关系来填补缺失值
    相关属性： Data_Value
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     Low_Confidence_Limit
    4.8     601
    5.4     593
    4.7     588
    4.9     574
    5.6     561
           ... 
    99.5      1
    99.6      1
    98.4      1
    97.8      1
    99.4      1
    Name: count, Length: 991, dtype: int64 
    
    替换缺失值后频数统计
     Low_Confidence_Limit
    37.341956    70009
    4.800000       601
    5.400000       593
    4.700000       588
    4.900000       574
                 ...  
    99.500000        1
    99.600000        1
    98.400000        1
    97.800000        1
    99.400000        1
    Name: count, Length: 992, dtype: int64 
    
    相关属性： Data_Value_Alt
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     Low_Confidence_Limit
    4.8     601
    5.4     593
    4.7     588
    4.9     574
    5.6     561
           ... 
    99.5      1
    99.6      1
    98.4      1
    97.8      1
    99.4      1
    Name: count, Length: 991, dtype: int64 
    
    替换缺失值后频数统计
     Low_Confidence_Limit
    37.341956    70009
    4.800000       601
    5.400000       593
    4.700000       588
    4.900000       574
                 ...  
    99.500000        1
    99.600000        1
    98.400000        1
    97.800000        1
    99.400000        1
    Name: count, Length: 992, dtype: int64 
    
    相关属性： Low_Confidence_Limit
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     Low_Confidence_Limit
    4.8     601
    5.4     593
    4.7     588
    4.9     574
    5.6     561
           ... 
    99.5      1
    99.6      1
    98.4      1
    97.8      1
    99.4      1
    Name: count, Length: 991, dtype: int64 
    
    替换缺失值后频数统计
     Low_Confidence_Limit
    32.736785    70009
    4.800000       601
    5.400000       593
    4.700000       588
    4.900000       574
                 ...  
    99.500000        1
    99.600000        1
    98.400000        1
    97.800000        1
    99.400000        1
    Name: count, Length: 992, dtype: int64 
    
    相关属性： High_Confidence_Limit
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     Low_Confidence_Limit
    4.8     601
    5.4     593
    4.7     588
    4.9     574
    5.6     561
           ... 
    99.5      1
    99.6      1
    98.4      1
    97.8      1
    99.4      1
    Name: count, Length: 991, dtype: int64 
    
    替换缺失值后频数统计
     Low_Confidence_Limit
    42.244436    70009
    4.800000       601
    5.400000       593
    4.700000       588
    4.900000       574
                 ...  
    99.500000        1
    99.600000        1
    98.400000        1
    97.800000        1
    99.400000        1
    Name: count, Length: 992, dtype: int64 
    
    一、将缺失部分剔除
    剔除缺失值前的数据集⼤⼩:  (214462,)
    剔除缺失值后的数据集⼤⼩:  (144453,)
    剔除缺失值后缺失值统计： 0
    
    二、通过属性的相关关系来填补缺失值
    相关属性： Data_Value
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     High_Confidence_Limit
    6.5    350
    5.8    349
    6.1    340
    5.9    339
    7.5    338
          ... 
    2.8      2
    1.4      2
    1.7      1
    1.5      1
    1.6      1
    Name: count, Length: 986, dtype: int64 
    
    替换缺失值后频数统计
     High_Confidence_Limit
    37.341956    70009
    6.500000       350
    5.800000       349
    6.100000       340
    5.900000       339
                 ...  
    2.800000         2
    1.400000         2
    1.700000         1
    1.500000         1
    1.600000         1
    Name: count, Length: 987, dtype: int64 
    
    相关属性： Data_Value_Alt
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     High_Confidence_Limit
    6.5    350
    5.8    349
    6.1    340
    5.9    339
    7.5    338
          ... 
    2.8      2
    1.4      2
    1.7      1
    1.5      1
    1.6      1
    Name: count, Length: 986, dtype: int64 
    
    替换缺失值后频数统计
     High_Confidence_Limit
    37.341956    70009
    6.500000       350
    5.800000       349
    6.100000       340
    5.900000       339
                 ...  
    2.800000         2
    1.400000         2
    1.700000         1
    1.500000         1
    1.600000         1
    Name: count, Length: 987, dtype: int64 
    
    相关属性： Low_Confidence_Limit
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     High_Confidence_Limit
    6.5    350
    5.8    349
    6.1    340
    5.9    339
    7.5    338
          ... 
    2.8      2
    1.4      2
    1.7      1
    1.5      1
    1.6      1
    Name: count, Length: 986, dtype: int64 
    
    替换缺失值后频数统计
     High_Confidence_Limit
    32.736785    70009
    6.500000       350
    5.800000       349
    6.100000       340
    5.900000       339
                 ...  
    2.800000         2
    1.400000         2
    1.700000         1
    1.500000         1
    1.600000         1
    Name: count, Length: 987, dtype: int64 
    
    相关属性： High_Confidence_Limit
    替换缺失值前的数据集⼤⼩:  (214462,)
    替换缺失值后的数据集⼤⼩:  (214462,)
    将缺失部分剔除后缺失值统计： 0 个
    替换缺失值前频数统计
     High_Confidence_Limit
    6.5    350
    5.8    349
    6.1    340
    5.9    339
    7.5    338
          ... 
    2.8      2
    1.4      2
    1.7      1
    1.5      1
    1.6      1
    Name: count, Length: 986, dtype: int64 
    
    替换缺失值后频数统计
     High_Confidence_Limit
    42.244436    70009
    6.500000       350
    5.800000       349
    6.100000       340
    5.900000       339
                 ...  
    2.800000         2
    1.400000         2
    1.700000         1
    1.500000         1
    1.600000         1
    Name: count, Length: 987, dtype: int64 
    
    三、通过数据对象之间的相似性来填补缺失值


代码仓库地址：https://github.com/LiukaSawaY/DMW3
