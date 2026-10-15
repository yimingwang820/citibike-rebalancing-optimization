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

大多数脚本使用固定的
D:\BikeSharing_OR_Project 路径。
迁移电脑需要调整路径常量。

注意：这不是已在新电脑完整验证的一键工作流。
复现同一结果需要原始订单和冻结中间数据。
公开仓库仅保留展示文档、结果图和小型汇总CSV，原始数据及大型中间结果被Git忽略。
当前展示文件的生成辅助脚本也未公开，不能将阅读现有图表等同于完整重建全部结果。
程序重命名的静态语法及逻辑检查，不代表全量求解或仿真回归已通过。
