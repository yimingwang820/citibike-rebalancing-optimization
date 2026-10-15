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

运行顺序：

Day1数据审计
→ Day2站点修复及库存
→ Day3/4优化建模
→ Day5事件回放
→ Day6跨日期验证
→ Day7交付。

主要训练验证脚本：

- day6_train_preposition.py
- day6_holdout_validation.py
- day6_weekend_train.py
- day6_weekend_validate.py

全部代码位于01_CODE目录。

大多数脚本使用固定的
D:\BikeSharing_OR_Project 路径。
迁移电脑需要调整路径常量。

注意：这不是已在新电脑完整验证的一键工作流。
复现同一结果需要原始订单和冻结中间数据。
