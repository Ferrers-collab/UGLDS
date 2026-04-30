# 建模代码说明

本目录直接读取 `../yuchuli/预处理结果/` 中的预处理产物，不重复清洗原始附件。`core.py` 已支持中文目录名与部分解压工具产生的 `#U9884...` 转义目录名。

## 目录结构

- `core.py`：共享数据层、需求拆分、路径评估、成本核算、路线解码
- `problem1.py`：问题 1 静态调度求解器（GA 风格排序优化 + 2-opt + MC 评估）
- `problem2.py`：问题 2 限行双目标求解器（NSGA-II 风格非支配筛选 + 熵权 TOPSIS）
- `problem3.py`：问题 3 动态事件重调度（快速基线修复 / 贪婪插入 / 限时 LNS 改善）
- `run.py`：统一命令行入口

## 运行方式

在项目根目录执行：

```bash
python -m model.run --problem all
```

只跑单问：

```bash
python -m model.run --problem 1
python -m model.run --problem 2
python -m model.run --problem 3
python -m model.run --problem 3 --p3-benchmark
```

输出默认写到 `model/results/`。

默认参数与 `run.py` 一致（见下），可在命令行覆盖，例如 `--p1-generations 200`。

- 问题 1：`--p1-population` 默认 60，`--p1-generations` 默认 120
- 问题 2：`--p2-population` 默认 60，`--p2-generations` 默认 120
- 问题 3：`--p3-iterations` 默认 8，`--p3-time-limit` 默认 8 秒；benchmark 每场景默认 3 次
- 项目已包含 `model/results/problem1/baseline_solution.pkl`，默认会优先复用问题 1 基线；如需强制重算，使用 `--recompute-p1`

## 建模实现细节

### 1. 数据直接复用预处理结果

- 客户属性：`customer_attributes.csv`
- 距离矩阵：`distance_matrix_clean.csv`
- 速度分布：`speed_profile.csv`
- 车队参数：`vehicle_fleet.csv`

### 2. 需求拆分

真实预处理数据中有不少客户总需求超过单车容量，因此代码先按最大车型能力自动拆分成多个虚拟配送任务，再进入路径优化。解码前会调用 `consolidate_split_order`：同一客户的多笔子任务在任务序列中**连续成段**（按 `split_index`），避免被其他客户任务拆散。

### 2.1 解码阶段路线评分（与主目标一致）

`route_selection_score` 仅使用：路线 `total_cost`（已含题给各项费用与不可行大罚）+ `tradeoff_lambda × carbon_kg`，并对每条非空路线加 **1e-6 × 该车型启动成本** 的极小正项，用于在成本几乎相同时略偏好更少分段；**不再使用**无题目/文献依据的经验常数。

### 3. 静态问题

问题 1 采用“任务序列 + 动态分段解码”的思路：

- 先对配送任务序列做启发式遗传搜索
- 再用动态规划把序列切分为若干条路径
- 路径内部做 2-opt 微调
- 对最优候选再做蒙特卡洛速度评估

### 4. 政策问题

问题 2 在问题 1 共享解码器基础上增加：

- 燃油车 8:00-16:00 禁入绿色区的硬约束
- 成本与碳排放双目标评价
- 非支配排序与拥挤度筛选
- 代表解的熵权 TOPSIS 选优

### 5. 动态问题

问题 3 默认基于问题 1 的基线方案自动生成一组示例事件，按**策略紧急度**、扰动度与任务规模判断：

- 小扰动且新增任务少：贪婪插入
- 大扰动或大规模未完成任务：`fast_baseline_repair` 秒级快速修复
- 需要进一步改善时：在 `--p3-time-limit` 限额内执行 LNS 风格重优化

输出中同时给出 **策略紧急度（用于分支）** 与 **事件影响度（与论文表一致，如取消=0.9）**，避免“论文写高影响、代码却因同一数值无法触发贪婪”的不一致。

### 2.2 多问题流水线

- `python -m model.run --problem all` 时：先求问题 1 静态解；问题 2 的对比基线为该问题 1 解；**问题 3 的基线重调度**也使用**同一份**问题 1 解（在问题 2 之后不替换为 Pareto 代表解，以保证“动态以静态方案为底稿”的语义）。

## 当前假设

以下是代码里明确补充的工程化假设：

- 静态调度默认只对有正需求的客户建模，零需求客户保留给动态事件模块使用
- 绿色区口径以附件坐标为准：距市中心≤10km 得到 15 个绿色区客户，其中 12 个有正需求；题面“30个”与附件不一致时，以附件数据为准
- 启动成本按当日实际启用物理车辆计，同一车辆多趟配送不重复计启动成本
- 速度分布表未覆盖 6:00 前和 17:00 后，代码将其外推为最近邻时段速度分布
- 动态模块把“未完成任务”抽象为扰动时刻之后的增量重优化集合，用于生成可复现实验代码

## 建议

如果你后面要写论文结果表，可直接运行：

```bash
python -m model.run --problem all
```

若只验证第三问实时调度与 benchmark，建议运行：

```bash
python -m model.run --problem 3 --p3-benchmark
```

然后直接从 `model/results/problem1/`、`problem2/`、`problem3/` 中取路线表、停靠表和汇总表。


### 运行时看不到输出怎么办

当前版本已经加入进度日志。建议使用：

```bash
python -u -m model.run --problem all
```

`-u` 用于关闭输出缓冲，能让进度提示立即显示。若终端仍然长时间停在问题1或问题2，说明正在执行遗传算法/多目标进化算法；可临时降低参数快速测试：

```bash
python -u -m model.run --problem all --p1-population 20 --p1-generations 20 --p2-population 20 --p2-generations 20
```

如需关闭日志，添加 `--quiet`。
