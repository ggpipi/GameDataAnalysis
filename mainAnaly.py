import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示和负号显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # Windows系统字体设置
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from datetime import datetime, timedelta

def new_file():
    # 设置随机种子保证可重复性
    np.random.seed(42)

    # 模拟1000个用户的12000条登录数据
    # 生成1万行以上的数据
    n_rows = 12000
    n_unique = 1000  # 最大不重复ID数量

    # ======================
    # 1. 基础字段生成
    # ======================
    # 1. 先生成1000个不重复的ID作为基础池
    unique_ids = np.random.choice(np.arange(10000, 20000), size=n_unique, replace=False)

    data = {
        # 用户ID：10000~19999之间的随机用户，允许重复
        "user_id": np.random.choice(unique_ids, size=n_rows, replace=True),

        # 登录时间：过去30天内的随机时间（使用列表生成式[]）
        "login_time": [
            (datetime.now() - timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )).strftime("%Y-%m-%d %H:%M:%S")
            for _ in range(n_rows)
        ],

        # 是否付费：20%的用户有付费行为
        "is_paid": np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),

        # 当前等级：正态分布（10~50级）
        "current_level": np.clip(np.random.normal(30, 10, n_rows), 1, 50).astype(int),

        # 登录时长：指数分布（分钟）
        "login_duration": np.clip(np.random.exponential(30, n_rows), 1, 1440).astype(int)
    }

    # ======================
    # 2. 付费金额生成（对数正态分布）
    # ======================
    # 生成付费金额（非付费用户金额为0）
    amount = np.round(np.random.lognormal(mean=2.5, sigma=1.2, size=n_rows), 2)
    data["amount"] = np.where(data["is_paid"] == 1, amount, 0)

    # ======================
    # 3. 新增payment_time列（仅付费用户有值）
    # ======================
    data["payment_time"] = [
        (datetime.strptime(login_time, "%Y-%m-%d %H:%M:%S") +
         timedelta(minutes=np.random.randint(1, login_duration))  # 随机增加1~login_duration-1分钟
         ).strftime("%Y-%m-%d %H:%M:%S")
        if (is_paid == 1 and login_duration > 1) else np.nan  # 仅对付费用户且登录时长>1分钟的生成时间
        for login_time, is_paid, login_duration in zip(data["login_time"], data["is_paid"], data["login_duration"])
    ]

    # ======================
    # 4. 分类字段生成（更新后的渠道列表）
    # ======================
    # 更新后的渠道来源（带权重）
    channels = ['Google Ads', 'Facebook', 'Organic', 'Apple Search Ads', 'Tik Tok', 'Unity Ads']
    data["channel"] = np.random.choice(
        channels,
        size=n_rows,
        p=[0.35, 0.25, 0.15, 0.1, 0.1, 0.05]  # 渠道权重
    )

    # 设备类型
    devices = ['iOS', 'Android', 'PC']
    data["device"] = np.random.choice(
        devices,
        size=n_rows,
        p=[0.5, 0.45, 0.05]  # 移动端占95%
    )

    # ======================
    # 5. 创建DataFrame并保存
    # ======================
    df = pd.DataFrame(data)

    # 添加10%的随机缺失值（login_time和payment_time不设缺失）
    cols_with_nan = ['amount', 'current_level', 'login_duration']
    for col in cols_with_nan:
        df.loc[df.sample(frac=0.1).index, col] = np.nan

    # 保存为CSV
    df.to_csv("game_logs.csv", index=False)
    print(f"生成成功！数据维度：{df.shape}")
    print("\n渠道分布统计:")
    print(df['channel'].value_counts(normalize=True))

if 0 == 1:
    new_file()
    print("新增数据")
else:
    print("不重新生成数据")

#######################################
# 1. 数据加载与清洗
#######################################
# 加载数据
df = pd.read_csv("game_logs.csv")

# 缺失值处理
df['amount'] = df['amount'].fillna(0)  # 付费金额缺失值填充为0
# 也可以用np.where填充缺失值
# df['amount'] = np.where(pd.isna(df['amount']), 0, df['amount'])

df = df.dropna(subset=['login_time'])  # 删除登录时间为空的行

# 检查是否存在同一用户在同一分钟重复登录的记录, keep=False为标记所有重复行
duplicates = df.duplicated(subset=['user_id', 'login_time'], keep=False).sum()
print(f"发现{duplicates}条同一用户同一分钟的重复登录记录")

# 转换登录时间为日期和小时
df['login_date'] = pd.to_datetime(df['login_time']).dt.date
df['login_hour'] = pd.to_datetime(df['login_time']).dt.hour
# 也可以用np.datetime64转换时间格式
# df['login_time'] = np.array(df['login_time'], dtype='datetime64[s]')
# df['login_date'] = df['login_time'].astype('datetime64[D]')  # 提取日期
# df['login_hour'] = df['login_time'].astype('datetime64[h]').astype(int) % 24  # 提取小时

# 用np.unique统计唯一用户数
unique_users = np.unique(df['user_id'])
print(f"唯一用户数量: {len(unique_users)}")

#######################################
# 2. 用户分层分析
#######################################
# 按付费金额将用户分为四档：非付费用户（0）、微氪（0-50）、中氪（50-200）、大R（>200）。
# 这里应该按照用户的总付费来分档，而不是每行数据分档
bins = [-1, 0, 50, 200, np.inf]
labels = ['非付费', '微氪', '中氪', '大R']
# 计算用户总付费并分层
user_total_payment = df.groupby('user_id')['amount'].sum().reset_index()
# 也可以用np.sum计算用户总付费（性能低）
# user_total_payment = df.groupby('user_id')['amount'].agg(np.sum).reset_index()
user_total_payment['pay_tier'] = pd.cut(user_total_payment['amount'], bins=bins, labels=labels)
# 下面是根据df的每行数据分档，是错的
# df['pay_tier'] = pd.cut(df['amount'], bins=bins, labels=labels)

# 用np.mean计算ARPU和ARPPU
arpu = np.mean(user_total_payment['amount'])
arppu = np.mean(user_total_payment[user_total_payment['amount'] > 0]['amount'])
print(f"ARPU: {arpu:.2f}, ARPPU: {arppu:.2f}")

# 用np.percentile分析分布
print(f"中位数: {np.percentile(user_total_payment['amount'], 50):.2f}")
print(f"90分位数: {np.percentile(user_total_payment['amount'], 90):.2f}")

#######################################
# 3. 用户活跃度分析
#######################################

# 用np.histogram分析登录时长分布
duration = df['login_duration'].values
# 计算各区间频数及区间边界
hist, edges = np.histogram(duration, bins=[0, 10, 30, 60, 1440])
# 等效的pandas实现
# df['duration_group'] = pd.cut(
#     df['login_duration'],
#     bins=[0, 10, 30, 60, 1440],
#     labels=['0-10分', '10-30分', '30-60分', '1-24小时']
# )
# dist = df['duration_group'].value_counts().sort_index()
print("登录时长分布:", dict(zip(edges, hist)))

# 用np.logical_and筛选高活跃高付费用户
high_active = np.logical_and(
    df.groupby('user_id')['login_date'].transform('nunique') > 7,
    df.groupby('user_id')['amount'].transform('sum') > 200
)
high_value_users = df[high_active]

# 统计高价值用户数量
print(f"高价值用户数: {len(high_value_users['user_id'].unique())}")

# 分析特征（示例：平均等级和登录时长）
print("平均等级:", high_value_users['current_level'].mean())
print("平均登录时长:", high_value_users['login_duration'].mean())

# 与非高价值用户对比付费率
normal_users = df[~df.index.isin(high_value_users.index)]
print("高价值用户平均付费:", high_value_users['amount'].mean())
print("普通用户平均付费:", normal_users['amount'].mean())

# 绘制高价值用户的付费金额分布
plt.hist(high_value_users.groupby('user_id')['amount'].sum(), bins=50, alpha=0.7)
plt.title("高价值用户付费金额分布")
plt.xlabel("金额")
plt.ylabel("用户数")
plt.show()

# 计算日活跃用户(DAU)
dau = df.groupby('login_date')['user_id'].nunique()
dau.plot(title='日活跃用户数(DAU)', figsize=(12, 6))
plt.xlabel('日期')
plt.ylabel('用户数')
plt.grid(True)
plt.show()

# 计算小时活跃用户分布
hourly_active = df['login_hour'].value_counts().sort_index() # 修改成按照index（0-24）排序
hourly_active.plot(kind='bar', title='用户活跃时段分布', figsize=(12, 6)) #柱状图
plt.xlabel('小时')
plt.ylabel('登录次数')
plt.xticks(rotation=0) #调整横坐标标签旋转角度
plt.grid(True)
plt.show()


#######################################
# 4. 付费行为分析
#######################################
# 筛选付费用户
paid_users = df[df['is_paid'] == 1]

# 付费用户渠道和设备分布分析
channel_dist = paid_users['channel'].value_counts(normalize=True)
device_dist = paid_users['device'].value_counts(normalize=True)

print("\n付费用户渠道分布:")
print(channel_dist)
print("\n付费用户设备分布:")
print(device_dist)

# 计算各渠道ARPPU(平均每付费用户收入)
arppu = paid_users.groupby('channel')['amount'].mean()
arppu.plot(kind='bar', title='各渠道ARPPU对比', figsize=(12, 6))
plt.xlabel('渠道')
plt.ylabel('ARPPU')
plt.xticks(rotation=0, ha='center')
plt.grid(True)
plt.show()


#######################################
# 5. 用户留存与流失分析
#######################################
# 用户流失标记
# 假设数据周期为7天，若用户最后登录时间早于第7天，标记为流失（is_churned=1）。
last_login = df.groupby('user_id')['login_date'].max()
# 这里要先reset_index后再merge才能比较，否则last_login的index是id，而df的index是0-31，index类型不匹配无法赋值
last_login_df = last_login.reset_index(name='last_login')
df = pd.merge(df, last_login_df, on='user_id', how='left')

cutoff_date = df['login_date'].max() - pd.Timedelta(days=7)
df['is_churned'] = (df['last_login'] < cutoff_date).astype(int)

# 流失用户特征分析
# 对比流失与非流失用户的平均关卡进度、登录时长。
churned = df[df['is_churned'] == 1]
non_churned = df[df['is_churned'] == 0]

print("\n流失用户平均特征:")
print(churned[['current_level', 'login_duration']].mean())
print("\n非流失用户平均特征:")
print(non_churned[['current_level', 'login_duration']].mean())


#######################################
# 6. 渠道效果评估
#######################################
def calculate_retention(merged_df, days):
    """计算指定天数的留存率"""
    # 提取所有用户的n日登录记录，标记符合条件的登录记录（首次登录后的n日）
    col_name = f'is_{days}_day'
    merged_df[col_name] = (merged_df['login_date'] - merged_df['first_login_date']) == pd.Timedelta(days=days)

    # 对每个用户，检查是否有任意一条记录满足n日登录
    user_retention = merged_df.groupby('user_id')[col_name].any().astype(int)
    user_retention = user_retention.reset_index().rename(columns={col_name: f'retained_day{days}'})

    # 合并用户渠道信息（如需分渠道统计）
    # 假设每个用户的渠道信息唯一（取第一条记录）
    user_channel = df.drop_duplicates('user_id')[['user_id', 'channel']]
    return pd.merge(user_retention, user_channel, on='user_id')

# 合并首次登录日期
first_login = df.groupby('user_id')['login_date'].min().reset_index(name='first_login_date')
merged = pd.merge(df, first_login, on='user_id')

# 计算次日和7日留存
user_retention_day1 = calculate_retention(merged, 1)
user_retention_day7 = calculate_retention(merged, 7)

# 整体留存率
overall_retention_day1 = user_retention_day1['retained_day1'].mean()
overall_retention_day7 = user_retention_day7['retained_day7'].mean()

print(f"\n整体次日留存率: {overall_retention_day1:.2%}")
print(f"整体7日留存率: {overall_retention_day7:.2%}")

channel_retention_day1 = user_retention_day1.groupby('channel')['retained_day1'].mean()
channel_retention_day7 = user_retention_day7.groupby('channel')['retained_day7'].mean()

print("\n分渠道次日留存率:")
print(channel_retention_day1)
print("\n分渠道7日留存率:")
print(channel_retention_day7)

# 用np.corrcoef计算相关性
corr_matrix = np.corrcoef(
    df.groupby('channel')['amount'].mean(),
    df.groupby('channel')['login_duration'].mean()
)
print(f"付费金额与登录时长的相关性: {corr_matrix[0,1]:.10f}")

# 用np.argmax找最佳渠道（返回最大值所在索引）
best_channel = df.groupby('channel')['amount'].sum().index[np.argmax(df.groupby('channel')['amount'].sum())]
# 也可以用pandas原生函数
# best_channel = df.groupby('channel')['amount'].sum().idxmax()
print(f"付费金额最高渠道: {best_channel}")


# 渠道ROI分析
# 假设已知各渠道投放成本，计算渠道的付费用户获取成本（CPA）和LTV（平均每个用户创造收益）。
# 假设渠道成本数据存储在channel_cost.csv，这是按渠道聚合的总成本（无日期维度）
cost_df = pd.read_csv('channel_cost.csv')
merged_cost = pd.merge(
    paid_users.groupby('channel')['user_id'].nunique(),
    cost_df,
    on='channel'
)
merged_cost['CPA'] = merged_cost['cost'] / merged_cost['user_id']

# 将付费总额转换为DataFrame，记得要reset成df格式才能merge
channel_revenue = paid_users.groupby('channel')['amount'].sum().reset_index(name='total_revenue')
# 合并两个DataFrame（确保merged_cost包含channel列）
merged_LTV = pd.merge(channel_revenue, merged_cost, on='channel', how='left')
# 计算LTV
merged_LTV['LTV'] = merged_LTV['total_revenue'] / merged_LTV['user_id']
print("\n渠道LTV分析:")
print(merged_LTV[['channel', 'LTV']])


#######################################
# 7. 数据可视化与报告
#######################################
# # 付费用户分布可视化
# # 绘制付费用户等级（微氪、中氪、大R）的占比饼图。
plt.figure(figsize=(8, 8))
user_total_payment['pay_tier'].value_counts(normalize=True).plot.pie(autopct='%.1f%%')
plt.title('付费用户分层占比')
plt.ylabel('')
plt.show()

# 渠道效果对比
# 用柱状图展示各渠道的ARPPU和留存率。
fig, ax1 = plt.subplots(figsize=(12, 6))# 增加宽度

# fig.subplots_adjust(bottom=0.3)          # 底部留更多空间
ax1.bar(
    channel_retention_day1.index,
    channel_retention_day1.values,
    color='blue',
    alpha=0.6,
    label='次日留存率'
)
ax1.set_ylabel('次日留存率', color='blue')
ax1.tick_params(axis='y', colors='blue')

ax2 = ax1.twinx()
ax2.plot(
    arppu.index,
    arppu.values,
    color='red',
    marker='o',
    label='ARPPU'
)
ax2.set_ylabel('ARPPU', color='red')
ax2.tick_params(axis='y', colors='red')

# 自动合并图例（无需handles）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('各渠道留存率与ARPPU对比')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(True)
plt.show()

#######################################
# 8. 高级分析
#######################################
# 1. 用np.polyfit拟合线性回归
x = df['current_level'].values
y = df['amount'].values
# 2. 清理无效值
mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
x_clean = x[mask]
y_clean = y[mask]
coefficients = np.polyfit(x_clean, y_clean, deg=1)
print(f"等级与付费金额关系: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}")

# 找出真正从未付费的用户（所有日志中is_paid均为0）
true_non_payers = df.groupby('user_id')['is_paid'].max() == 0
true_non_payer_ids = true_non_payers[true_non_payers].index.values

# 2. 检查非付费用户数量是否足够
if len(true_non_payer_ids) < 100:
    raise ValueError(f"非付费用户不足{100}人，仅有{len(true_non_payer_ids)}人")

# 3. 随机抽样（确保不重复）
sampled_ids = np.random.choice(true_non_payer_ids, size=100, replace=False)

# 4. 获取这些用户的所有日志（包括他们未付费和付费的记录）
sampled_logs = df[df['user_id'].isin(sampled_ids)]
print(f"抽样用户数: {len(sampled_ids)}")
print(f"抽样日志数: {len(sampled_logs)}")

# 3. 用np.diff计算登录时长变化率
daily_duration = df.groupby(['user_id', 'login_date'])['login_duration'].mean().unstack()
# 处理日期连续性
all_dates = pd.date_range(df['login_date'].min(), df['login_date'].max())
daily_duration = daily_duration.reindex(columns=all_dates)

duration_changes = np.diff(daily_duration.values, axis=1) # 这里是返回index是id，前后2天时间的差
print("每日登录时长变化率描述:", {
    'mean': np.nanmean(duration_changes), # 这里是对全局求均值，无视行列，无视所有nan
    'std': np.nanstd(duration_changes)
})
# pandas非等效实现，这里是求2次均值，会受到数据nan的影响（不推荐）
# changes = daily_duration.diff(axis=1).iloc[:, 1:] # 这里iloc[:, 1:]是因为差分的时候要去掉第一列nan
# print(changes.mean().mean(), changes.stack().std())

# 绘制变化率分布直方图
plt.hist(duration_changes[~np.isnan(duration_changes)], bins=50)
plt.xlabel("每日登录时长变化（分钟）")
plt.ylabel("频次")
plt.title("用户活跃度变化分布")
plt.axvline(0, color='red', linestyle='--')  # 标记零变化线
plt.show()

# 找出单日时长下降超过30分钟的用户
large_drop = (duration_changes < -30).any(axis=1) # 已隐式忽略nan
drop_users = daily_duration.index[large_drop]

# 检查变化率与次日留存的关系
df_changes = pd.DataFrame({
    'user_id': daily_duration.index,
    'avg_change': np.nanmean(duration_changes, axis=1)
})
merged = pd.merge(df_changes, user_retention_day1, on='user_id')
print(merged[['avg_change', 'retained_day1']].corr())

# 预测流失风险
from sklearn.linear_model import LogisticRegression

duration_changes = np.nan_to_num(duration_changes, nan=0)  # 将NaN替换为0

# 特征：近期时长变化均值
X = np.nanmean(duration_changes[:, -3:], axis=1).reshape(-1, 1)  # 最近3天变化均值
y = df.groupby('user_id')['is_churned'].max().values  # 假设已有流失标签
# X和y都是通过groupby生成的，因此他们的user_id应该是对齐的，可以直接输入到model

model = LogisticRegression(class_weight='balanced').fit(X, y) # 样本不均衡，添加 class_weight='balanced' 参数

# 查看模型系数
print("截距 (intercept):", model.intercept_)
print("系数 (coefficients):", model.coef_)
# 系数 (coefficients) 表示每个特征对结果的影响方向和大小
# 正系数表示该特征增加会提高正类 (1) 的概率
# 负系数表示该特征增加会降低正类 (1) 的概率

# 计算特征重要值（OR值）
odds_ratio = np.exp(model.coef_)
print("OR 值 (指数化系数):", odds_ratio)
# 指数化系数 (OR 值) 解释为：特征每增加 1 个单位，结果发生的几率 (odds) 将乘以 OR 值
# OR > 1 表示正相关，OR < 1 表示负相关

# 模型性能评估
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X)
print("准确率:", accuracy_score(y, y_pred))
print("混淆矩阵:\n", confusion_matrix(y, y_pred))
print("分类报告:\n", classification_report(y, y_pred))

# 计算 ROC 曲线和 AUC
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_scores = model.predict_proba(X)[:, 1]
fpr, tpr, thresholds = roc_curve(y, y_scores)
print("AUC 值:", roc_auc_score(y, y_scores))

# 绘制 ROC 曲线
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 计算置信区间 (使用 statsmodels 更准确)
import stats  as sm

# 使用 statsmodels 获取更详细的统计信息
logit_model = sm.Logit(y, sm.add_constant(X))
result = logit_model.fit(method='firth') # 对于小样本 (n < 1000)，考虑使用 Firth 惩罚逻辑回归 ( statsmodels 的 method='firth 选项)
print(result.summary())
# 会显示 p 值、置信区间等统计指标
# 可以判断特征是否统计显著

# 模型预测概率分析
# 查看预测概率分布
pd.DataFrame(model.predict_proba(X), columns=['class_0', 'class_1']).describe()

# 根据概率阈值调整预测
custom_threshold = 0.4  # 自定义阈值
custom_pred = (model.predict_proba(X)[:, 1] >= custom_threshold).astype(int)

# 检查模型校准
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y, y_scores, n_bins=10)
plt.plot(prob_pred, prob_true, 's-')
plt.plot([0, 1], [0, 1], 'k:')
plt.xlabel('预测概率')
plt.ylabel('实际比例')
plt.title('校准曲线')
plt.show()

