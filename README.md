# 城市绿色物流配送调度

华中杯大学生数学建模挑战赛参赛作品

## 项目概述

本项目针对城市绿色物流配送调度问题，建立了一套完整的优化求解系统。问题涵盖静态路径优化、政策限行双目标优化以及实时动态重调度三个层面。

## 目录结构

```
huazhongcup/
├── model/                     # 优化模型求解器
│   ├── core.py              # 共享数据层、路径评估、成本核算
│   ├── problem1.py           # 问题1：静态调度（GA + 2-opt + 蒙特卡洛评估）
│   ├── problem2.py           # 问题2：限行双目标（NSGA-II + 熵权TOPSIS）
│   ├── problem3.py           # 问题3：动态事件重调度
│   ├── problem3_benchmark.py # 问题3基准测试
│   └── run.py                # 统一命令行入口
├── preprocess/                # 数据预处理模块
│   ├── 附件/                 # 原始数据文件
│   ├── 预处理结果/            # 清洗后的数据文件
│   └── 预处理图/             # 数据可视化图表
└── docs/                     # 模型文档
    ├── 第一问模型建立final.md
    ├── 第二问模型建立final.md
    └── 第三问模型建立final.md
```

## 数据说明

预处理模块读取以下原始附件：

- 客户坐标信息.xlsx
- 订单信息.xlsx
- 时间窗.xlsx
- 距离矩阵.xlsx

输出清洗后的数据文件：

- `customer_attributes.csv` - 客户属性表
- `customer_demand_clean.csv` - 客户需求清洗结果
- `distance_matrix_clean.csv` - 距离矩阵
- `speed_profile.csv` - 速度时段分布
- `vehicle_fleet.csv` - 车队参数
- `orders_clean.csv` - 订单清洗结果

## 快速开始

### 运行全部问题

```bash
python -m model.run --problem all
```

### 运行单问

```bash
python -m model.run --problem 1
python -m model.run --problem 2
python -m model.run --problem 3
```

### 问题3基准测试

```bash
python -m model.run --problem 3 --p3-benchmark
```

### 加速测试（降低迭代次数）

```bash
python -u -m model.run --problem all --p1-population 20 --p1-generations 20 --p2-population 20 --p2-generations 20
```

## 主要参数

| 问题  | 参数                 | 默认值 |
| --- | ------------------ | --- |
| 问题1 | `--p1-population`  | 60  |
| 问题1 | `--p1-generations` | 120 |
| 问题2 | `--p2-population`  | 60  |
| 问题2 | `--p2-generations` | 120 |
| 问题3 | `--p3-iterations`  | 8   |
| 问题3 | `--p3-time-limit`  | 8秒  |

### 基线缓存控制

- `--reuse-p1`：若存在缓存则优先复用
- `--recompute-p1`：强制重新求解问题1

## 问题描述

### 问题1：静态调度

在给定客户需求、时间窗和车队条件下，求解单日配送路径优化问题，最小化总运营成本。

算法采用遗传算法 + 2-opt局部搜索 + 蒙特卡洛速度评估。

### 问题2：政策限行双目标

考虑燃油车8:00-16:00禁止进入绿色区的政策约束，以成本和碳排放为双目标进行优化。

采用NSGA-II非支配排序 + 熵权TOPSIS选优。

### 问题3：动态事件重调度

基于问题1的基线方案，处理订单取消、新增订单等动态事件，实现实时重调度。

根据扰动规模自适应选择：贪婪插入 / 快速基线修复 / LNS重优化。

## 需求拆分机制

当客户总需求超过单车容量时，系统自动拆分为多个虚拟配送任务。解码阶段保证同一客户的子任务在路线中连续，避免被其他客户任务拆散。

## 多问题流水线

- 问题1求解完成后，结果缓存至 `model/results/problem1/baseline_solution.pkl`
- 问题2使用问题1的解作为对比基线
- 问题3的重调度底稿同样使用问题1的解（而非问题2的Pareto代表解）

## 输出结果

运行结果保存在 `model/results/` 目录下：

- `problem1/` - 问题1路线表、停靠表
- `problem2/` - 问题2 Pareto前沿解集
- `problem3/` - 问题3动态调度方案

## 当前假设

以下为本代码明确的工程化假设：

1. 静态调度默认只对有正需求的客户建模，零需求客户保留给动态事件模块使用
2. 绿色区口径以附件坐标为准：距市中心≤10km得到15个绿色区客户，其中12个有正需求
3. 启动成本按当日实际启用物理车辆计，同一车辆多趟配送不重复计启动成本
4. 速度分布表未覆盖6:00前和17:00后，代码将其外推为最近邻时段速度分布
5. 动态模块把"未完成任务"抽象为扰动时刻之后的增量重优化集合

## 数据预处理

如需重新运行数据预处理：

```bash
cd huazhongcup/preprocess/预处理结果
python datawashing.py
```

## 成本构成

总成本 = 启动成本 + 能耗成本 + 碳排放成本 + 时间窗惩罚

- 启动成本：400元/辆
- 燃油能耗：0.0025v² - 0.2554v + 31.75 (L/100km)，油价7.61元/L
- 电耗：0.0014v² - 0.12v + 36.19 (kWh/100km)，电价1.64元/kWh
- 碳排放：2.547 kg/L（燃油）/ 0.501 kg/kWh（电动），碳价0.65元/kg
- 早到等待：20元/h
- 迟到惩罚：50元/h
