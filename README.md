# Citi Bike 共享单车站点库存预调拨优化

基于 Python、Gurobi MILP 和订单级离散事件仿真，
研究共享单车站点的车辆库存配置、借车拒绝与归还满桩问题。

## 项目内容

- 处理2026年4月历史骑行数据，修复站点ID别名。
- 构建25站研究区域，保留订单OD与时间顺序。
- 建立单周期及多周期MILP，优化整数调拨和库存配置。
- 加入固定任务成本、容量约束和调拨预算。
- 构建30分钟满桩等待的事件驱动仿真。
- 分别训练工作日和周末的预调拨方案。
- 利用跨日期实验分析策略的收益与局限。

## 主要实验结果

### 工作日

训练得到80辆、6条有向关系的固定预调拨方案。

在7个验证工作日中：

- 模拟归还超时：1650 → 1370
- 归还超时下降：16.97%
- 借车拒绝：228 → 236
- 7天的归还超时均有所减少

### 周末

原工作日方案直接迁移到两个周末时，
归还超时由14次增加到42次。

周末专用W12/W24方案可以降低部分模拟等待时间，
但未降低归还超时，同时增加了借车拒绝。

三折留一周末代理验证也显示策略稳定性不足。

## 实验数据划分

- 工作日训练：2026-04-01至04-20，共14个工作日。
- 周末训练：4月4、5、11、12、18、19日。
- 开发诊断：4月21日。
- 跨日期验证：4月22日至30日。

4月25–26日已用于周末模型诊断，
不能再视为完全未接触的最终盲测日期。

## 项目文件

- `01_CODE/`：23个按功能命名的正式程序，当前仍保留在同一目录
- `03_SHOWCASE/MODEL_AND_LIMITATIONS.md`：模型说明
- `03_SHOWCASE/EXPERIMENTS.md`：实验结果
- `03_SHOWCASE/*.png`：结果图

## 研究限制

初始库存为合成情景，站点容量使用非同期GBFS代理值。
调拨假设在每日回放前完成，尚未考虑卡车载重与运输时间。

历史已完成订单被用于反事实仿真；
服务改善不等于Citi Bike真实运营收益。

## 环境与复现

- 环境：Python 3.12、NumPy、Pandas、Matplotlib、Gurobi。
- 安装依赖：`python -m pip install -r requirements.txt`
- 使用Gurobi需要有效许可证。
- 原始订单：https://s3.amazonaws.com/tripdata/202604-citibike-tripdata.zip
- GBFS站点信息：https://gbfs.citibikenyc.com/gbfs/en/station_information.json

注意：本实验采用已冻结的非同期GBFS容量快照。
重新下载当前快照不保证复现相同容量与优化结果。

本地原始数据放在 `00_DATA/`；
中间数据与结果保存在 `02_RESULTS/`。

主流程：

1. 订单审计、站点ID修复、研究区域选择与初始库存构建。
2. 构建工作日小时流量，提取开发日订单并生成事件仿真基线。
3. 求解工作日午夜预调拨方案，在开发日诊断并进行跨日期验证。
4. 独立训练周末方案，再进行三折代理验证与周末事件验证。

建议优先阅读：

- [weekday_preposition.py](01_CODE/weekday_preposition.py)：工作日优化模型；入口还会运行开发日仿真。
- [timeout_simulation.py](01_CODE/timeout_simulation.py)：30分钟超时事件仿真。
- [holdout_validation.py](01_CODE/holdout_validation.py)：冻结工作日方案的跨日期验证。
- [weekend_preposition.py](01_CODE/weekend_preposition.py)：周末12/24小时优化模型。
- [weekend_validation.py](01_CODE/weekend_validation.py)：三折重新求解及周末事件验证。

LP/MILP、固定成本和小时级多周期模型保留为对照实验，
其求解结果不是最终预调拨主流程的必要输入。
现有 `02_RESULTS/DAY*` 目录和冻结结果文件名保持不变。

多数脚本使用 `D:\BikeSharing_OR_Project` 绝对路径，
迁移到其他设备需要调整路径。

完整阶段依赖见
`03_SHOWCASE/REPRODUCIBILITY.md`。

## 解释限制

工作日模型使用前12小时训练数据；
周末W24使用全天24小时数据，且部分约束不同。

因此80辆与9辆的调拨规模差异，
不能全部归因于日期类型。

本项目的服务改善为反事实模拟结果，
不是实际运营收益。
