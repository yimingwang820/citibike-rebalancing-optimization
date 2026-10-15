# 项目环境与复现说明

1. Python 3.12。
2. 安装依赖：
   `python -m pip install -r requirements.txt`
3. 另需有效Gurobi许可证。
4. 原始订单：
   `00_DATA/202604-citibike-tripdata.zip`
5. 冻结GBFS站点快照：
   `00_DATA/station_information.json`

请保留当时使用的容量快照；
重新下载当天GBFS可能导致结果不同。

## 程序位置与运行方式

23个正式程序已按功能命名，全部仍位于 `01_CODE/`，尚未迁移到 Python 子包。
以下命令均从项目根目录执行，采用 `python 01_CODE/程序名.py`。
输入和输出仍使用原有 `02_RESULTS/DAY*` 目录、CSV/JSON 文件名；程序改名不改变数据口径。

这些命令会处理数据、求解或回放，并写入相应结果目录，不能作为无副作用的检查命令。
已有冻结结果时不要直接顺序重跑。需要重新实验时，应先为相关脚本配置独立输出位置；
多数入口目前尚无通用输出目录参数。

## 1. 数据准备

```powershell
python 01_CODE/audit_trip_data.py --zip 00_DATA/202604-citibike-tripdata.zip --metadata 00_DATA/station_information.json --output 02_RESULTS
python 01_CODE/normalize_station_ids.py --input-dir 02_RESULTS --metadata 00_DATA/station_information.json --output-dir 02_RESULTS/DAY2_ID_FIXED
python 01_CODE/select_study_area.py --input-dir 02_RESULTS/DAY2_ID_FIXED --metadata 00_DATA/station_information.json --output-dir 02_RESULTS/DAY2_A_FIXED
python 01_CODE/build_inventory_scenarios.py
python 01_CODE/build_weekday_profiles.py
python 01_CODE/extract_trip_events.py
```

审计输出需显式指定为 `02_RESULTS`，不能沿用默认 `audit_results`。
选站入口需显式读取修复目录并写入 `DAY2_A_FIXED`，不能沿用其默认 `DAY2_A`。
选站表包含多个候选区域，下游仍按种子 `5980.10` 选择25站。
订单提取入口默认针对开发日2026-04-21；跨日期验证会自行提取相应日期的订单。

## 2. 基线、工作日训练与跨日期验证

```powershell
python 01_CODE/event_baseline.py
python 01_CODE/timeout_simulation.py
python 01_CODE/weekday_preposition.py
python 01_CODE/holdout_validation.py
```

- `event_baseline.py`：四种库存情景，无30分钟退出规则。
- `timeout_simulation.py`：读取基线汇总作为核对依据，运行30分钟超时仿真。
- `weekday_preposition.py`：读取工作日小时流量训练，再运行开发日仿真；需要已有超时基线结果。
- `holdout_validation.py`：读取冻结的工作日方案和前后库存，扫描订单并回放4月22–30日；不重新优化。

## 3. 周末扩展

```powershell
python 01_CODE/weekend_preposition.py
python 01_CODE/weekend_validation.py
```

周末训练仅使用6个训练周末日的小时流量。周末验证需要已有周末候选、工作日冻结方案和
`DAY6_HOLDOUT/holdout_daily_comparison.csv`；它会重新求解三折模型，并回放4月25–26日。
这两个日期已用于诊断，不能称为最终盲测。

## 4. 可选对照与诊断

| 程序（均在01_CODE） | 用途与前置条件 |
|---|---|
| `single_period_lp.py` | 单小时LP；依赖修复小时表、选站和库存；支持 `--check-data-only`。 |
| `compare_lp_milp.py` | 同数据LP/MILP比较；依赖同上；支持 `--check-data-only`。 |
| `fixed_charge_milp.py` | 固定启用成本及关系数上限对照；依赖同上。 |
| `fixed_cost_sensitivity.py` | 调用固定成本模型进行多次求解。 |
| `hourly_balance_milp.py` | 四小时聚合平衡模型；依赖工作日小时profile。 |
| `audit_hourly_feasibility.py` | 读取上一项结果，检查库存平衡和先借后还解释下的缺口。 |
| `strict_order_milp.py` | 四小时分阶段模型；复用聚合模型的数据加载器，不要求先求解聚合模型。 |
| `minimal_event_simulation.py` | 独立合成教学示例；事件优先级与正式仿真不同。 |
| `service_metrics.py` | 从已有无超时基线CSV计算服务指标，不回放。 |
| `wait_duration_diagnostics.py` | 重放无超时事件以诊断等待时长，并与已有基线核对。 |
| `development_day_effects.py` | 读取工作日训练后的开发日结果，解释站点/小时收益，不回放。 |

这些对照模型的求解结果不属于最终预调拨主流程的必需输入。

## 复现边界

项目根目录由 `01_CODE/project_paths.py` 根据代码位置确定，不依赖启动目录。
可在启动 Python 前设置环境变量 `CITIBIKE_PROJECT_ROOT` 指向另一份项目根目录；
所有模块在导入时绑定同一根目录，导入后改变环境变量不会更新已绑定路径，应重启进程。

CLI 显式相对路径仍相对于当前工作目录。为保留已有默认行为，
`audit_trip_data.py` 的 `audit_results` 和 `select_study_area.py` 的相对路径默认值保持原样；
这两个入口从其他目录启动时须显式传入绝对输入/输出路径（详见复现说明）。
默认运行仍可能覆盖原结果，检查时不要执行正式入口。

例如，在 PowerShell 中先设置 `$env:CITIBIKE_PROJECT_ROOT = 'E:\BikeSharing_OR_Project'`，
再启动 Python；不需要改动源码。取消覆盖可使用 `Remove-Item Env:CITIBIKE_PROJECT_ROOT`。
该设置改变共享根目录，不会重解释 CLI 显式传入的相对路径。
从其他工作目录执行数据准备入口时，应把上述命令中的脚本、输入和输出路径全部写为绝对路径。
Day 1 的 `audit_results` 与下游输入不衔接、Day 2 的 `DAY2_A` 与下游 `DAY2_A_FIXED`
不衔接，是保留的历史默认行为；是否统一这些默认值应另行决定。

轻量检查：从项目根目录运行 `python -B -m unittest discover -s tests -v`。
测试需要已安装 requirements 中的依赖和本地阶段1 Git 检查点；不需要原始订单或有效求解许可证。
检查覆盖跨工作目录、根目录覆盖、模块导入副作用以及全部23个程序的静态一致性；
所有测试缓存写入系统临时目录，不调用正式入口，不代表全量仿真回归通过。

## 日期参数与兼容边界

单日函数 `event_baseline.read_inputs`、`event_baseline.simulate`、
`timeout_simulation.replay`、`wait_duration_diagnostics.replay_waits`
支持关键字参数 `date="YYYY-MM-DD"`。省略时仍使用开发日 `2026-04-21`；
多日调用应每次显式传入日期，不再给其他模块的 `DATE` 赋值。
`read_inputs` 仍读取原开发日文件路径，传日期只改变该文件的当日统计口径，
不会自动提取另一日订单；跨日期数据仍须由原提取流程提供。

`holdout_validation.extract_holdout`、`holdout_validation.reconcile`、
`holdout_validation.main` 和 `weekend_validation.event_evaluation`
支持关键字参数 `dates`，元素为上述日期字符串。
调用者的日期顺序和重复项保留；仅用于成员筛选的本地集合去重，不新增排序。
省略时仍分别采用4月22–30日、4月25–26日；训练日期和交叉验证折划分保持原样。
`DATE_SET` 仅保留兼容快照，动态提取不再依赖修改该全局变量。

CLI不变。显式传日期不改变输入输出目录或既有冻结对照要求，
因此不能把自定义日期直接作为安全试跑方式；验证入口仍可能覆盖结果。
日期测试使用临时人工订单，与本地阶段2A检查点实现对照，
覆盖日期隔离、跨午夜、同刻优先级、FIFO、30分钟边界、提取对账及异常退出。
测试依赖本地阶段1、阶段2A的Git历史；浅克隆需要补齐这两个检查点。

注意：这不是已在新电脑完整验证的一键工作流。
复现同一结果需要原始订单和冻结中间数据。
公开仓库仅保留展示文档、结果图和小型汇总CSV，原始数据及大型中间结果被Git忽略。
当前展示文件的生成辅助脚本也未公开，不能将阅读现有图表等同于完整重建全部结果。
程序重命名的静态语法及逻辑检查，不代表全量求解或仿真回归已通过。
