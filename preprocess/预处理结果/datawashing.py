# -*- coding: utf-8 -*-
"""
城市绿色物流配送调度 —— 数据预处理（最终版）
严格遵循预处理模块方案，并已适配官方补充说明：
  1. 绿色区客户数以实际计算为准（15个）
  2. 速度分布参数采用修正后的标准差：顺畅 σ=0.1, 一般 σ=5.2, 拥堵 σ=4.7
清洗数据保存至根目录，可视化图表保存至“预处理结果”文件夹。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# ==================== 路径设置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
VIS_DIR = os.path.join(BASE_DIR, "预处理结果")
if not os.path.exists(VIS_DIR):
    os.makedirs(VIS_DIR)

# ==================== 1. 读取原始数据 ====================
print("正在读取原始数据...")
df_orders = pd.read_excel(os.path.join(DATA_DIR, "订单信息.xlsx"))
df_distance = pd.read_excel(os.path.join(DATA_DIR, "距离矩阵.xlsx"), index_col=0)
df_coords = pd.read_excel(os.path.join(DATA_DIR, "客户坐标信息.xlsx"))
df_timewin = pd.read_excel(os.path.join(DATA_DIR, "时间窗.xlsx"))

# ==================== 2. 缺失值处理（严格按照模块方案） ====================
# 2.1 清理列名，确保数值类型
df_orders.columns = [col.strip() for col in df_orders.columns]
df_orders['重量'] = pd.to_numeric(df_orders['重量'], errors='coerce')
df_orders['体积'] = pd.to_numeric(df_orders['体积'], errors='coerce')

# 2.2 同客户均值填充（核心步骤）
df_orders['重量'] = df_orders.groupby('目标客户编号')['重量'].transform(
    lambda x: x.fillna(x.mean())
)
df_orders['体积'] = df_orders.groupby('目标客户编号')['体积'].transform(
    lambda x: x.fillna(x.mean())
)

# 2.3 若某客户所有订单均缺失，则组均值仍为NaN，使用全局均值填充
global_mean_weight = df_orders['重量'].mean()
global_mean_volume = df_orders['体积'].mean()
df_orders['重量'] = df_orders['重量'].fillna(global_mean_weight)
df_orders['体积'] = df_orders['体积'].fillna(global_mean_volume)

print("缺失值处理完成。")

# ==================== 3. 数据转换与特征工程 ====================

# 3.1 订单聚合为客户总需求
customer_demand = df_orders.groupby('目标客户编号').agg(
    总重量_kg=('重量', 'sum'),
    总体积_m3=('体积', 'sum')
).reset_index()
customer_demand.rename(columns={'目标客户编号': '客户编号'}, inplace=True)

# 3.2 绿色区标记（距离市中心 ≤ 10km）
df_customers_coords = df_coords[df_coords['类型'] == '客户'].copy()
df_customers_coords['距市中心_km'] = np.sqrt(
    df_customers_coords['X (km)']**2 + df_customers_coords['Y (km)']**2
)
df_customers_coords['绿色区'] = (df_customers_coords['距市中心_km'] <= 10).astype(int)

# 3.3 时间窗数值化（转为分钟数）
def time_to_minutes(t_str):
    h, m = map(int, str(t_str).split(':'))
    return h * 60 + m

df_timewin['最早_min'] = df_timewin['开始时间'].apply(time_to_minutes)
df_timewin['最晚_min'] = df_timewin['结束时间'].apply(time_to_minutes)

# 3.4 构建完整客户属性表（强制统一客户编号为 int，避免合并失败）
df_attributes = df_customers_coords[['ID', 'X (km)', 'Y (km)', '距市中心_km', '绿色区']].copy()
df_attributes.rename(columns={'ID': '客户编号'}, inplace=True)

df_attributes['客户编号'] = df_attributes['客户编号'].astype(int)
customer_demand['客户编号'] = customer_demand['客户编号'].astype(int)
df_timewin['客户编号'] = df_timewin['客户编号'].astype(int)

# 合并需求与时间窗
df_attributes = df_attributes.merge(customer_demand, on='客户编号', how='left')
df_attributes = df_attributes.merge(df_timewin[['客户编号', '最早_min', '最晚_min']], on='客户编号', how='left')

# 将无订单记录的客户需求置为 0（这些客户当日无配送需求）
missing_demand = df_attributes['总重量_kg'].isnull()
if missing_demand.sum() > 0:
    missing_ids = df_attributes.loc[missing_demand, '客户编号'].tolist()
    print(f"⚠ 以下 {len(missing_ids)} 个客户在订单表中无记录，将其总需求置为 0：{missing_ids}")
    df_attributes['总重量_kg'] = df_attributes['总重量_kg'].fillna(0)
    df_attributes['总体积_m3'] = df_attributes['总体积_m3'].fillna(0)

df_attributes.sort_values('客户编号', inplace=True)
df_attributes.reset_index(drop=True, inplace=True)

# 3.5 速度时段参数（已按官方补充说明修正 σ 值）
speed_profile = pd.DataFrame([
    ['拥堵1', 480, 540, 9.8, 4.7],
    ['一般1', 540, 600, 35.4, 5.2],
    ['顺畅1', 600, 690, 55.3, 0.1],
    ['拥堵2', 690, 780, 9.8, 4.7],
    ['顺畅2', 780, 900, 55.3, 0.1],
    ['一般2', 900, 1020, 35.4, 5.2]
], columns=['时段', '开始_min', '结束_min', 'mu_kmph', 'sigma_kmph'])

# 3.6 车辆参数表
vehicle_fleet = pd.DataFrame([
    ['燃油车1', 3000, 13.5, 60, 400, '燃油'],
    ['燃油车2', 1500, 10.8, 50, 400, '燃油'],
    ['燃油车3', 1250, 6.5,  50, 400, '燃油'],
    ['新能源车1', 3000, 15,  10, 400, '新能源'],
    ['新能源车2', 1250, 8.5,  15, 400, '新能源']
], columns=['车型', '载重_kg', '容积_m3', '数量', '启动成本_元', '动力类型'])

# ==================== 4. 异常值检测（Z-score） ====================
from scipy import stats
z_weight = np.abs(stats.zscore(df_attributes['总重量_kg']))
z_volume = np.abs(stats.zscore(df_attributes['总体积_m3']))
outliers_weight = df_attributes['客户编号'][z_weight > 3]
outliers_volume = df_attributes['客户编号'][z_volume > 3]
if len(outliers_weight) > 0:
    print(f"重量异常客户（Z-score>3）: {outliers_weight.tolist()}")
if len(outliers_volume) > 0:
    print(f"体积异常客户（Z-score>3）: {outliers_volume.tolist()}")

# ==================== 5. 保存清洗后数据至根目录 ====================
print("保存清洗后数据到根目录...")
customer_demand.to_csv(os.path.join(BASE_DIR, 'customer_demand_clean.csv'), index=False, encoding='utf-8-sig')
df_attributes.to_csv(os.path.join(BASE_DIR, 'customer_attributes.csv'), index=False, encoding='utf-8-sig')
df_distance.to_csv(os.path.join(BASE_DIR, 'distance_matrix_clean.csv'), encoding='utf-8-sig')
speed_profile.to_csv(os.path.join(BASE_DIR, 'speed_profile.csv'), index=False, encoding='utf-8-sig')
vehicle_fleet.to_csv(os.path.join(BASE_DIR, 'vehicle_fleet.csv'), index=False, encoding='utf-8-sig')
df_orders.to_csv(os.path.join(BASE_DIR, 'orders_clean.csv'), index=False, encoding='utf-8-sig')
print("所有清洗后数据文件已保存至根目录。")

# ==================== 6. 可视化（保存至 预处理结果 文件夹） ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 客户空间分布
fig, ax = plt.subplots(figsize=(12, 10))
circle = plt.Circle((0, 0), 10, color='green', fill=False, linestyle='--', linewidth=2)
ax.add_patch(circle)
green_in = df_attributes[df_attributes['绿色区'] == 1]
green_out = df_attributes[df_attributes['绿色区'] == 0]
ax.scatter(green_in['X (km)'], green_in['Y (km)'], c='green', s=50, label=f'区内客户 ({len(green_in)})')
ax.scatter(green_out['X (km)'], green_out['Y (km)'], c='orange', marker='^', s=50, label=f'区外客户 ({len(green_out)})')
center = df_coords[df_coords['类型'] == '配送中心']
ax.scatter(center['X (km)'], center['Y (km)'], c='red', marker='*', s=300, label='配送中心')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_title('客户空间分布及绿色配送区（半径10km）')
ax.legend()
ax.set_aspect('equal')
plt.savefig(os.path.join(VIS_DIR, '客户空间分布.png'), dpi=300)
plt.close()

# 需求分布
fig, ax = plt.subplots(figsize=(14, 6))
demand_sorted = df_attributes.sort_values('总重量_kg', ascending=False)
colors = ['green' if g == 1 else 'orange' for g in demand_sorted['绿色区']]
ax.bar(range(1, 99), demand_sorted['总重量_kg'], color=colors, alpha=0.7)
ax.set_xlabel('客户编号（按重量降序）')
ax.set_ylabel('总需求重量 (kg)')
ax.set_title('各客户需求重量分布')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.7, label='绿色区内'),
                   Patch(facecolor='orange', alpha=0.7, label='绿色区外')]
ax.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, '需求分布.png'), dpi=300)
plt.close()

# 时间窗分布
fig, ax = plt.subplots(figsize=(14, 10))
tw_sorted = df_attributes.sort_values('最早_min')
for i, (_, row) in enumerate(tw_sorted.iterrows()):
    ax.barh(i, row['最晚_min'] - row['最早_min'], left=row['最早_min'],
            height=0.6, color='skyblue', edgecolor='black')
ax.set_xlabel('时间 (分钟，从0:00起)')
ax.set_ylabel('客户')
ax.set_title('客户时间窗分布')
ax.set_xticks(range(480, 1260, 60))
ax.set_xticklabels([f'{h}:00' for h in range(8, 21)])
ax.set_xlim(420, 1320)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, '时间窗分布.png'), dpi=300)
plt.close()
print("可视化图表已保存至：", VIS_DIR)

# ==================== 7. 数据完整性验证（全软警告） ====================
print("\n========== 数据完整性验证 ==========")
assert len(df_attributes) == 98, "客户数量错误"
print(f"✓ 客户数量：98")

green_count = df_attributes['绿色区'].sum()
if green_count != 30:
    print(f"⚠ 绿色区内客户数为 {green_count}（官方说明：以实际计算为准）。")

# 检查缺失值
if df_attributes.isnull().sum().sum() > 0:
    print("⚠ 客户属性表仍存在缺失值！")
else:
    print("✓ 客户属性表无缺失值")

# 需求为0的客户
zero_demand = df_attributes[df_attributes['总重量_kg'] == 0]
if len(zero_demand) > 0:
    print(f"⚠ 注意：有 {len(zero_demand)} 个客户总需求为0（无订单记录），客户编号：{zero_demand['客户编号'].tolist()}")
else:
    print("✓ 所有客户总重量 > 0")

if df_orders['重量'].isnull().sum() > 0:
    print("⚠ 订单重量仍有缺失")
else:
    print("✓ 订单重量无缺失")
if df_orders['体积'].isnull().sum() > 0:
    print("⚠ 订单体积仍有缺失")
else:
    print("✓ 订单体积无缺失")

# 时间窗合法性
invalid_tw = df_attributes[df_attributes['最早_min'] >= df_attributes['最晚_min']]
if len(invalid_tw) > 0:
    print(f"⚠ 时间窗不合法客户：{invalid_tw['客户编号'].tolist()}")
else:
    print("✓ 所有时间窗合法")

# 距离矩阵
dist_mat = df_distance.values
assert dist_mat.shape == (99, 99)
assert np.allclose(dist_mat, dist_mat.T), "距离矩阵不对称"
assert np.all(np.diag(dist_mat) == 0), "对角线非0"
assert (dist_mat >= 0).all(), "存在负值"
print("✓ 距离矩阵正确")

print("\n数据预处理成功完成！所有清洗数据已准备就绪。")
