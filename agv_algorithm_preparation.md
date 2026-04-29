# AGV 调度优化编码前准备说明

## 1. 目标

本文档用于在正式实现 `data_loader`、`scheduler_decoder`、`fitness_function`、GA 和 PSO 之前，统一算法可直接调用的数据结构、解码规则、适应度函数和测试场景，确保论文模型与程序实现口径一致。

---

## 2. 参数冻结表

### 2.1 固定参数

| 参数 | 符号 | 取值 | 说明 |
|---|---|---:|---|
| AGV 额定速度 | \(v_0\) | 0.5 m/s | 所有 AGV 统一速度 |
| 单位距离能耗 | \(\beta\) | \(5\times 10^{-4}\) | 每米消耗总电量万分之五 |
| 三车道数量 | \(N^{lane}\) | 3 | 存储区与生产区之间 |
| 单车道容量 | \(C^{lane}\) | 1 | 同一时刻一条车道仅允许 1 台 AGV |
| 通道总容量 | \(C^{corridor}\) | 3 | 三条车道并行 |
| 充电位数量 | \(N^{chg}\) | 10 | 采用显式给出的 10 个坐标 |
| 换道惩罚时间 | \(t^{lc}\) | 2 s | 发生车道切换时计入 |
| 工人干扰率 | \(\lambda_{prod}\) | 0.02 1/s | 生产区边扰动参数 |
| 拥堵敏感系数 | \(\alpha\) | 0.6 | 边拥堵修正参数 |
| 最低安全电量 | \(q^{min}\) | 2.0 | 统一采用额定电量对应的安全阈值 |
| 充电触发阈值 | \(q^{th}\) | 4.0 | 低于该值需插入充电任务 |
| 充电功率 | \(P^{ch}\) | 10 kW | 简化为常数充电功率 |

### 2.2 坐标冻结规则

1. **物料存储区**采用货位中心坐标公式：
   \[
   (x,y)=\bigl(4.2+3.2(c-1),\ 5.2+3.2(r-1)\bigr)
   \]
2. **成品存储区**采用货位中心坐标公式：
   \[
   (x,y)=\bigl(4.2+3.2(c-1),\ -3.0-3.2(r-1)\bigr)
   \]
3. **不良品区**固定为第 10 行，坐标为：
   \[
   (x,y)=\bigl(4.2+3.2(c-1),\ -31.8\bigr)
   \]
4. **工位节点**采用最新给定坐标，不再与早期版本混用。
5. **算法实现阶段**将每个工位或库存服务点视为一个独立服务节点，不再额外拆分“投放点节点”和“工位本体节点”。

---

## 3. 数据接口设计

## 3.1 节点表 `nodes`

每个节点建议采用如下字段：

| 字段名 | 类型 | 说明 |
|---|---|---|
| `node_id` | str | 节点唯一编号 |
| `node_type` | str | 节点类型，如 `storage`、`finish`、`defect`、`line`、`buffer`、`charger`、`junction` |
| `region` | str | 区域，如 `storage`、`finish`、`production`、`charging`、`road` |
| `x` | float | X 坐标 |
| `y` | float | Y 坐标 |
| `service_capable` | bool | 是否可作为任务服务节点 |

示例：

```python
{
    "node_id": "S_1_1",
    "node_type": "storage",
    "region": "storage",
    "x": 4.2,
    "y": 5.2,
    "service_capable": True,
}
```

## 3.2 边表 `edges`

每条有向边建议采用如下字段：

| 字段名 | 类型 | 说明 |
|---|---|---|
| `edge_id` | str | 边编号 |
| `from_node` | str | 起点 |
| `to_node` | str | 终点 |
| `distance` | float | 边长度（m） |
| `base_time` | float | 基础通行时间（s） |
| `energy_cost` | float | 通行能耗 |
| `capacity` | int | 边容量 |
| `edge_type` | str | 如 `storage_grid`、`finish_grid`、`main_road`、`lane`、`charger_link` |
| `is_production_edge` | bool | 是否位于生产区 |

示例：

```python
{
    "edge_id": "E_S_1_1__S_1_2",
    "from_node": "S_1_1",
    "to_node": "S_1_2",
    "distance": 3.2,
    "base_time": 6.4,
    "energy_cost": 0.0016,
    "capacity": 1,
    "edge_type": "storage_grid",
    "is_production_edge": False,
}
```

## 3.3 任务表 `tasks`

每个任务建议采用如下字段：

| 字段名 | 类型 | 说明 |
|---|---|---|
| `task_id` | str | 任务编号 |
| `task_type` | str | 如 `raw_material`、`finished_goods`、`defect_transfer` |
| `pickup_node` | str | 取货节点 |
| `drop_node` | str | 送货节点 |
| `qty` | float | 任务载荷 |
| `earliest_start` | float | 最早开始时间 |
| `latest_finish` | float | 最晚完成时间 |
| `pickup_service_time` | float | 取货服务时间 |
| `drop_service_time` | float | 卸货服务时间 |

说明：为了更方便编码，原论文中的 \(p_i\) 建议拆分为取货和卸货两个服务时间字段。

## 3.4 AGV 表 `agvs`

每台 AGV 建议采用如下字段：

| 字段名 | 类型 | 说明 |
|---|---|---|
| `agv_id` | str | AGV 编号 |
| `start_node` | str | 初始待命点或充电位 |
| `battery_init` | float | 初始电量 |
| `battery_max` | float | 最大电量 |
| `battery_threshold` | float | 充电触发阈值 |
| `battery_min` | float | 最低安全电量 |
| `charge_power` | float | 充电功率 |
| `load_capacity` | float | 最大载重 |
| `status` | str | 初始状态 |

---

## 4. 任务类型生成规则

为保证后续实验场景统一，建议将任务分为三类：

### 4.1 原料配送任务

\[
pickup_i\in V^S,\qquad drop_i\in V^{L1}\cup V^{L2}\cup V^{L3}\cup V^{L4}
\]

表示从物料存储区向生产工位配送原料。

### 4.2 成品转运任务

\[
pickup_i\in V^{L1}\cup V^{L2}\cup V^{L3}\cup V^{L4},\qquad drop_i\in V^F\cup V^B
\]

表示从工位或暂存区转运到成品区。

### 4.3 不良品转运任务

\[
pickup_i\in V^{L4}\cup V^B,\qquad drop_i\in V^D
\]

表示质检不合格品转入不良品区。

建议在测试场景中按如下比例生成任务：

- 原料配送：50%
- 成品转运：35%
- 不良品转运：15%

---

## 5. 解码器规则

## 5.1 调度解的编码解释

算法解码统一采用“双层编码”：

1. **任务分配向量**
   - 第 \(i\) 个任务分配给哪台 AGV。
2. **任务优先级向量**
   - 用于确定同一 AGV 上任务的执行顺序。

对每台 AGV，解码后的任务执行方式固定为：

\[
\text{当前位置}\rightarrow pickup_i \rightarrow drop_i \rightarrow pickup_j \rightarrow drop_j \rightarrow \cdots
\]

即每个任务以“完整闭环”的方式执行，不采用“批量取货后统一送货”的模式。

## 5.2 最短路调用规则

任意两个节点间的行驶成本通过最短路函数计算：

\[
D(a,b),\quad T^{base}(a,b),\quad E(a,b)
\]

其中：

- \(D(a,b)\)：最短路径距离；
- \(T^{base}(a,b)\)：最短路径基础时间；
- \(E(a,b)=\beta D(a,b)\)：最短路径基础能耗。

## 5.3 电量检查规则

在准备执行一个新任务前，对 AGV 当前电量进行检查。若当前电量不足以支持：

1. 从当前位置到取货点；
2. 从取货点到送货点；
3. 从送货点到最近充电位；

则必须先插入充电任务。

## 5.4 充电插入规则

算法初版采用如下简化规则：

1. 选择当前时刻**最近且可用**的充电位；
2. AGV 前往充电位后**充满电**再离开；
3. 充电完成后返回下一项待执行任务流程。

该规则在后续可扩展为部分充电策略，但初版统一采用“充满电策略”。

## 5.5 拥堵处理规则

初版解码器不采用细粒度连续仿真，而采用离散事件方式：

1. 记录 AGV 进入每条边的开始时间和结束时间；
2. 若两台或多台 AGV 在同一时段占用同一容量为 1 的边，则后进入的 AGV 产生等待；
3. 等待时间计入拥堵惩罚和实际完工时间。

三车道通道的处理方式为：

- 单条车道同一时刻最多 1 台 AGV；
- 三条车道整体最多 3 台 AGV；
- 若通道整体未满，优先选择预计最早可通行的车道。

## 5.6 工人干扰处理规则

为保证 GA 和 PSO 评价稳定，Day 1 至 Day 3 的实现采用**确定性近似**：

\[
\mathbb{E}[\xi_{uv}] = \frac{1}{\lambda_{prod}}
\]

因此对生产区边采用：

\[
t_{uv}^{act}=t_{uv}^{base}\,c_{uv}(t)+\frac{1}{\lambda_{prod}}
\]

第 4 天实验阶段再扩展为随机扰动敏感性分析。

---

## 6. 适应度函数与惩罚项

## 6.1 评价指标

统一评价以下指标：

1. 最大完工时间 \(T_{\max}\)
2. 总里程 \(D_{total}\)
3. 总能耗 \(E_{total}\)
4. 时间窗违约数或违约总量 \(L_{late}\)
5. 充电次数 \(N_{chg}\)
6. 拥堵等待时间 \(W_{cong}\)

## 6.2 适应度函数

建议初版采用单目标加惩罚形式：

\[
F=
\omega_1 T_{\max}
+\omega_2 E_{total}
+\omega_3 W_{cong}
+\omega_4 L_{late}
+\omega_5 P_{battery}
+\omega_6 P_{capacity}
+\omega_7 P_{load}
\]

其中：

- \(P_{battery}\)：电量违规惩罚；
- \(P_{capacity}\)：边容量或充电位容量违规惩罚；
- \(P_{load}\)：载重违规惩罚。

## 6.3 初始权重建议

为使不可行解显著劣于可行解，建议初始权重设置为：

| 项 | 权重 |
|---|---:|
| \(\omega_1\) | 1.0 |
| \(\omega_2\) | 50 |
| \(\omega_3\) | 20 |
| \(\omega_4\) | 500 |
| \(\omega_5\) | 2000 |
| \(\omega_6\) | 2000 |
| \(\omega_7\) | 1500 |

这些权重后续可在第 4 天实验阶段做敏感性分析。

## 6.4 违反项定义建议

### 电量违规惩罚

\[
P_{battery}=\sum_k \max\{0,\ q^{min}-q_k^{minpath}\}
\]

### 时间窗违规惩罚

\[
L_{late}=\sum_{i\in T}\max\{0,\ C_i-l_i\}
\]

### 容量违规惩罚

\[
P_{capacity}=\sum_{(u,v)\in E}\max\{0,\ n_{uv}(t)-C_{uv}\}
\]

### 载重违规惩罚

\[
P_{load}=\sum_{i\in T}\sum_{k\in A}\max\{0,\ qty_i-Q_k^{load}\}x_{ik}
\]

---

## 7. Day 1 最小测试场景规范

为验证 `data_loader`、`scheduler_decoder` 和 `fitness_function` 的正确性，建议先构造一个 2 台 AGV、6 个任务的小规模测试场景。

## 7.1 AGV 设置

- AGV 数量：2
- 初始位置：\(C_1, C_2\)
- 初始电量：满电
- 最大载重：统一为 50 kg

## 7.2 任务设置

建议 6 个任务分布如下：

1. 2 个原料配送任务：\(S \rightarrow L1/L2\)
2. 2 个成品转运任务：\(L1/L4 \rightarrow F\)
3. 2 个不良品转运任务：\(L4 \rightarrow D\)

## 7.3 人工调度方案验证

预设一个人工调度方案：

- AGV1 执行任务 \(1\rightarrow 3\rightarrow 5\)
- AGV2 执行任务 \(2\rightarrow 4\rightarrow 6\)

核心检查项：

1. 每台 AGV 的实际路径是否可回溯；
2. 每个任务的开始/结束时间是否可计算；
3. 电量递推是否正确；
4. 目标函数值是否可复现；
5. 插入充电任务后，调度结果是否仍然可解释。

---

## 8. Day 1 编码产出要求

在完成上述准备后，Day 1 的程序开发目标明确为：

### 8.1 `data_loader`

负责读取并构造：

- `nodes`
- `edges`
- `tasks`
- `agvs`
- 图结构 `graph`

### 8.2 `scheduler_decoder`

负责将一个给定调度方案解码为：

- 每台 AGV 的任务序列
- 每段行驶路径
- 每个任务的开始/完成时间
- 每台 AGV 的电量变化
- 充电插入记录

### 8.3 `fitness_function`

负责计算：

- 最大完工时间
- 总里程
- 总能耗
- 拥堵等待时间
- 时间窗违约量
- 惩罚项
- 最终 fitness

---

## 9. 与 GA / PSO 的接口约定

为方便后续直接接入 GA 和 PSO，统一约定：

1. 输入解码结构：
   ```python
   solution = {
       "assignment": [...],
       "priority": [...],
   }
   ```
2. 输出评价结构：
   ```python
   evaluation = {
       "fitness": ...,
       "makespan": ...,
       "total_distance": ...,
       "total_energy": ...,
       "late_penalty": ...,
       "charge_count": ...,
       "congestion_wait": ...,
       "agv_schedules": ...,
   }
   ```
3. GA 与 PSO 均调用同一套：
   - `decode_solution(solution, instance)`
   - `evaluate_solution(solution, instance)`

这样可以保证第 2 天和第 3 天在同一评价标准下公平对比。
