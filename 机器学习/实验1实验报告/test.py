#%%引入库函数
import pandas as pd
import seaborn as sb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as mp
import numpy as np
import time as systime
import datetime as dt
import string
from sklearn import preprocessing
import matplotlib.colors as colors
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

#设置中文字体
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']= False

#%%读取数据

train_path = r'D:\机器学习\旧金山犯罪率预测\train.csv'
test_path = r'D:\机器学习\旧金山犯罪率预测\test.csv'
sampleSubmission_path = r'D:\机器学习\旧金山犯罪率预测\sampleSubmission.csv'

train = pd.read_csv(train_path)
sampleSubmission = pd.read_csv(sampleSubmission_path)
test = pd.read_csv(test_path)

#%%查看数据情况
# 查看数据集类型
train.info()

# 查看数据集的缺失数据情况
'''
isnull的返回值：
如果是缺失值（NaN/None）则返回 True，
否则返回 False，最终得到一个与原 DataFrame 
形状相同的布尔型 DataFrame

然后把这个bool类型的DataFrame累加得到有多少true
'''
train.isnull().sum()

# 通过groupby().size()方法返回分组后的统计结果

cate_group = train.groupby(by='Category').size() 
cate_group

train.groupby(["Category", "PdDistrict"]).count()
#%%绘制柱状图
# 格式化索引，将每个单词首字母大写
cate_group.index = cate_group.index.map(string.capwords)

# 按值降序排序
cate_group.sort_values(ascending=False, inplace=True)

# 获取类别数量，用于配色
cat_num = len(cate_group)

# 绘制柱状图
cate_group.plot(
    kind='bar',
    logy=True,
    figsize=(15, 10),
    color=sns.color_palette('cool', cat_num)
)

# 设置图表标题
plt.title('No. of Crime types', fontsize=20)

# 显示图表
plt.show()


#%%hisplot类型柱状图绘制
#histplot绘制
cate_group = train.groupby(by='Category').size() 

train.groupby(["Category", "PdDistrict"]).count()
# 格式化索引，将每个单词首字母大写
cate_group.index = cate_group.index.map(string.capwords)


# 转换为DataFrame，方便seaborn处理
df = pd.DataFrame({
    'Crime Type': cate_group.index,
    'Count': cate_group.values
})

# 获取类别数量，用于配色
cat_num = len(cate_group)

# 使用histplot绘制（适合展示分布，这里模拟柱状图效果）
plt.figure(figsize=(15, 10))
sns.histplot(
    data=df,
    x='Crime Type',
    weights='Count',  # 用Count列的值作为权重
    discrete=True,    # 表示x是离散类别
    palette=sns.color_palette('cool', cat_num)
)



# 设置纵轴为对数刻度
# plt.yscale('log')

# 设置图表标题和标签
plt.title('No. of Crime types', fontsize=20)
plt.xlabel('Crime Type', fontsize=14)
plt.ylabel('Count', fontsize=14)

# 旋转x轴标签以防重叠
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

#%%绘制散点图
import matplotlib.pyplot as plt
def scatter(filter, title, hue, size):
    plt.figure(figsize=(19, 19))
    sb.scatterplot(
        data=train[filter],
        x='X', y='Y',
        alpha=0.6, 
        palette='tab10',
        hue=hue, 
        size=size
    )
    # 可选：注释或调整坐标轴限制
    # plt.xticks(np.arange(-123, -122))
    # plt.yticks(np.arange(37, 38 ))
    
    plt.title(title)
    # 优化图例位置（放在图内）
    plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))  

# 筛选数据集
coord_filter = ~(train["X"] > -121)
day_filter = train["DayOfWeek"] == "Wednesday"
scatter(day_filter & coord_filter, "Day Wednesday", "Category", "Category")
plt.show()
#%%保存地图(课件示例)
import folium 
# define the national map
national_map = folium.Map(location=[32.03,118.85], zoom_start=16)
# save national map
national_map.save('D:/机器学习/旧金山犯罪率预测/Njust.html')

#%%保存地图(山财燕山)
import folium 
# define the national map
tiles = 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7'
national_map = folium.Map(location=[36.64, 117.07], tiles=tiles, attr='高德-常规图', zoom_start=16)
# save national map
national_map.save('D:/机器学习/旧金山犯罪率预测/Njust_高德.html')

#%%生成train_data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

train_path = r'D:\机器学习\旧金山犯罪率预测\train.csv'
train = pd.read_csv(train_path)

'''
# 看各个特征的详细信息
#unique为23228，舍弃
s = pd.Series(train['Address'])
print(s.describe(), '\n')

# unique为10，保留
s = pd.Series(train['PdDistrict'])
print(s.describe(), '\n')

# unique为7(一周7天)，保留
s = pd.Series(train['DayOfWeek'])
print(s.describe(), '\n')
'''

# 处理 Dates 列，提取 Hour 特征
df = pd.DataFrame(train)
df['Dates'] = pd.to_datetime(df['Dates'])
df['Hour'] = df['Dates'].dt.hour  # 新增小时特征到 df 中


# 关键修改：将Hour转换为字符串类型，以便get_dummies进行独热编码
# 否则的话，就默认Hour为数值特征，不会进行正常的独热编码
df['Hour'] = df['Hour'].astype(str)
'''
# 小时特征 ，unique的大小为24，保留
s = pd.Series(df['Hour'])
x = s.unique()
print(x.size, '\n')
'''

# 选择要二值化的特征列
features_to_encode = ['DayOfWeek', 'PdDistrict', 'Hour']
# 二值化处理，使用 df 而不是 train
# prefix: 前缀内容 / prefix_sep:前缀与类别值之间的分隔符，默认"_"
encoded_features = pd.get_dummies(df[features_to_encode],prefix='',prefix_sep='')

# 按照数字顺序重新排列Hour列
# 首先获取所有Hour相关的列名
hour_columns = [col for col in encoded_features.columns if col.isdigit()]
# 转换为整数并排序
hour_columns_sorted = sorted(hour_columns, key=lambda x: int(x))
# 获取非Hour的列名
non_hour_columns = [col for col in encoded_features.columns if not col.isdigit()]
# 重新排列列
encoded_features = encoded_features[non_hour_columns + hour_columns_sorted]


# 查看二值化后的特征
print(encoded_features.head())

# 标签编码
cirme = preprocessing.LabelEncoder( )

# 创建LabelEncoder对象
le = preprocessing.LabelEncoder()
# 对犯罪类型进行编码
df['crime'] = le.fit_transform(df['Category'])

# 将特征与标签拼接
train_data = pd.concat([encoded_features, df['crime']], axis=1)


# 查看最终训练集的形状和前几行
print(f"训练集形状: {train_data.shape}")  # 应该是 (样本数, 41+1)
print("\n训练集前5行:")
print(train_data.head())

# 打印每一列的名字
print(train_data.columns.tolist()) 

# 将训练数据保存为Excel文件
# excel_path = r'D:\机器学习\旧金山犯罪率预测\train_processed.xlsx'  # 保存路径和文件名
# train_data.to_excel(excel_path, index=False)  # index=False表示不保存索引列

#%%生成test_data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

test_path = r'D:\机器学习\旧金山犯罪率预测\test.csv'
test = pd.read_csv(test_path)

# 处理 Dates 列，提取 Hour 特征
df = pd.DataFrame(test)
df['Dates'] = pd.to_datetime(df['Dates'])
df['Hour'] = df['Dates'].dt.hour  # 新增小时特征到 df 中


df['Hour'] = df['Hour'].astype(str)

# 选择要二值化的特征列
features_to_encode = ['DayOfWeek', 'PdDistrict', 'Hour']
# 二值化处理，使用 df 而不是 test
# prefix: 前缀内容 / prefix_sep:前缀与类别值之间的分隔符，默认"_"
encoded_features = pd.get_dummies(df[features_to_encode],prefix='',prefix_sep='')

# 按照数字顺序重新排列Hour列
# 首先获取所有Hour相关的列名
hour_columns = [col for col in encoded_features.columns if col.isdigit()]
# 转换为整数并排序
hour_columns_sorted = sorted(hour_columns, key=lambda x: int(x))
# 获取非Hour的列名
non_hour_columns = [col for col in encoded_features.columns if not col.isdigit()]
# 重新排列列
encoded_features = encoded_features[non_hour_columns + hour_columns_sorted]

# 查看二值化后的特征
print(encoded_features.head())


# 将特征与标签拼接
test_data = encoded_features


# 查看最终训练集的形状和前几行
print(f"训练集形状: {test_data.shape}")  # 应该是 (样本数, 41+1)
print("\n训练集前5行:")
print(test_data.head())

# 打印每一列的名字
print(test_data.columns.tolist()) 

#%% 划分train的训练集与验证集
from sklearn.model_selection import train_test_split

# 1. 先从 train_data 中分离出特征（X）与标签（y）
X = train_data.drop('crime', axis=1)
y = train_data['crime']               

# 2. 用 train_test_split 划分数据
#    test_size=0.3 表示 30% 数据用作验证集
#    random_state=42 保证每次划分一致
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. 将划分好的数据拼接回 DataFrame 形式，方便查看
#    训练集：X_train + y_train
train_split = pd.concat([X_train, y_train], axis=1)
#    验证集：X_val + y_val
val_split = pd.concat([X_val, y_val], axis=1)

# 4. 查看划分结果
print(f"原始样本数: {train_data.shape[0]}")
print(f"训练集样本数: {train_split.shape[0]}")
print(f"验证集样本数: {val_split.shape[0]}")

# 5. 预览前5行数据
print("\n训练集前5行：")
print(train_split.head())
print("\n验证集前5行：")
print(val_split.head())

#%% 线性回归分析(调用函数)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# 1. 创建线性回归模型实例
lr_model = LinearRegression()
# 2. 使用训练集训练模型
lr_model.fit(X_train, y_train)
# 3. 查看各项特征的系数
xishu = pd.DataFrame({
    'Feature': X_train.columns,
    'xishu': lr_model.coef_
})
# 按系数绝对值大小排序
xishu['Abs_xishu'] = xishu['xishu'].abs()
xishu = xishu.sort_values('Abs_xishu', ascending=False)
print("\n特征系数（按绝对值大小排序）：")
print(xishu[['Feature', 'xishu']])
# 4. 查看截距
print(f"\n线性回归模型截距: {lr_model.intercept_}")
# 5. 在验证集上评估模型性能
y_pred = lr_model.predict(X_val)
# 计算均方误差
#mse = mean_squared_error(y_val, y_pred)
# 计算R^2分数
#r2 = r2_score(y_val, y_pred)
#print(f"\n在验证集上的均方误差 (MSE): {mse:.4f}")
#print(f"在验证集上的R^2分数: {r2:.4f}")
# 6. 可视化系数
plt.figure(figsize=(15, 10))
# 只显示绝对值最大的20个特征的系数
top_n = 20
plt.barh(range(top_n), xishu['xishu'].head(top_n), color='skyblue')
plt.yticks(range(top_n), xishu['Feature'].head(top_n))
plt.xlabel('系数')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%% 运行线性回归并保存结果
result_path = r'D:\机器学习\旧金山犯罪率预测\linear_regression_results.xlsx'
xishu.to_excel(result_path, index=False)
print(f"\n线性回归结果已保存至: {result_path}")
#%% 计算预测成功概率（准确率）
from sklearn.metrics import accuracy_score
import numpy as np

y_pred_rounded = np.round(y_pred)

# 确保预测值在有效范围内（不小于0，不大于最大类别值）
min_category = 0
max_category = np.max(y_train)
y_pred_clamped = np.clip(y_pred_rounded, min_category, max_category)

y_pred_int = y_pred_clamped.astype(int)

# 计算准确率（预测正确的样本数占总样本数的比例）
accuracy = accuracy_score(y_val, y_pred_int)
# 计算精确率、召回率和F1分数
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_val, y_pred_int, average='macro', zero_division=0)
recall = recall_score(y_val, y_pred_int, average='macro', zero_division=0)
f1 = f1_score(y_val, y_pred_int, average='macro', zero_division=0)

print(f"预测准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"精确率 : {precision:.4f}")
print(f"召回率 : {recall:.4f}")
print(f"F1分数 : {f1:.4f}")

# 对于回归模型，我们也可以计算预测值与真实值的绝对误差和相对误差
absolute_errors = np.abs(y_pred - y_val)
relative_errors = absolute_errors / (y_val + 1e-10)  # 加小值避免除零

print(f"平均绝对误差: {np.mean(absolute_errors):.4f}")
print(f"中位数绝对误差: {np.median(absolute_errors):.4f}")
print(f"平均相对误差: {np.mean(relative_errors)*100:.2f}%")
print(f"预测值在真实值±10%范围内的样本比例: {np.mean(np.abs(y_pred - y_val) <= y_val*0.1)*100:.2f}%")
print(f"预测值在真实值±1%范围内的样本比例: {np.mean(np.abs(y_pred - y_val) <= 1)*100:.2f}%")


#%% 计算 Dates、DayOfWeek、PdDistrict 特征的重要性

from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 训练一个随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 获取每个特征的重要性得分
feature_importances = pd.Series(rf.feature_importances_, index=X_train.columns)

# 按原始特征分组求和（因为每个类别特征被独热编码成多列）
feature_groups = {
    'DayOfWeek': [col for col in X_train.columns if col in ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']],
    'PdDistrict': [col for col in X_train.columns if col in ['BAYVIEW','CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK','RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN']],
    'Hour': [col for col in X_train.columns if col.isdigit()]
}

# 计算每组的平均重要性
group_importance = {group: feature_importances[cols].mean() for group, cols in feature_groups.items()}

# 转成 DataFrame 方便展示
group_importance_df = pd.DataFrame(list(group_importance.items()), columns=['Feature Group', 'Mean Importance'])
group_importance_df.sort_values(by='Mean Importance', ascending=False, inplace=True)

print("各特征组的重要性：")
print(group_importance_df)

# 选出最重要的两个特征组
top_features = group_importance_df.head(2)['Feature Group'].tolist()
print("\n最重要的两个特征组为：", top_features)



#%%训练GBM模型

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

# 1. 选择前面特征重要性分析中最重要的两个特征（DayOfWeek 与 PdDistrict）
selected_features = []
selected_features += [col for col in X_train.columns if col in ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']]
selected_features += [col for col in X_train.columns if col in ['BAYVIEW','CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK','RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN']]

X_train_sel = X_train[selected_features]
X_val_sel = X_val[selected_features]

# 2. 建立GBM模型
gbm = GradientBoostingClassifier(
    n_estimators=100,   # 树的数量
    learning_rate=0.1,  # 学习率
    max_depth=3,        # 每棵树的最大深度
    random_state=42
)

# 3. 模型训练
gbm.fit(X_train_sel, y_train)

# 4. 在验证集上预测类别概率
y_val_pred_proba = gbm.predict_proba(X_val_sel)

# 5. 计算log_loss
val_logloss = log_loss(y_val, y_val_pred_proba)
print(f"验证集 log_loss: {val_logloss:.4f}")

# 6. （可选）在测试集上预测并导出结果
test_pred_proba = gbm.predict_proba(test_data[selected_features])

submission = pd.DataFrame(test_pred_proba, columns=le.classes_)
submission.to_csv('D:/机器学习/旧金山犯罪率预测/gbm_submission.csv', index=False)

print("GBM 模型预测完成，结果已保存为 gbm_submission.csv")


#%%预测概率


import numpy as np
import pandas as pd

# 预测测试集概率（每个类别的概率）
test_pred_proba = gbm.predict_proba(test_data[selected_features])

# 获取类别标签（le 是 LabelEncoder）
class_labels = list(le.classes_)

# 构建结果DataFrame
test_result = pd.DataFrame(test_pred_proba, columns=class_labels)

# 按标签名从小到大排序列（确保一致性）
test_result = test_result[sorted(test_result.columns)]

# 检查每行概率是否和为1（四舍五入后输出）
row_sums = test_result.sum(axis=1).round(6)
assert np.allclose(row_sums, 1.0), "存在概率和不为1的样本！"

# 找出每行概率最大的类别（Predicted_Crime）
test_result['Predicted_Crime'] = test_result.idxmax(axis=1)

# 添加Id列（从0开始递增）
test_result['Id'] = range(len(test_result))

# 输出结果
print(test_result.head())

# 保存到本地
test_result.to_csv('D:/机器学习/旧金山犯罪率预测/test_result.csv', index=False)
print("test_result.csv 已保存！")