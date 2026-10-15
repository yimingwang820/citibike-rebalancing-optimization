
# -*- coding: utf-8 -*-
"""
Day 4-A：构建 Citi Bike 多周期模型的小时数据

使用：
1. 修复后的站点小时流量
2. 已选定的25站主实验区域
3. 2026-04-01 至 2026-04-20 训练工作日

输出：
- train_station_day_hour.csv
- train_hourly_station_profiles.csv
- train_hourly_area_profiles.csv
- day4_data_audit.json

直接运行，不需要命令行参数。
"""

from pathlib import Path
import json

import pandas as pd
import numpy as np


# ==================================================
# 1. 路径和参数
# ==================================================

ROOT = Path(r"D:\BikeSharing_OR_Project")

SEED = "5980.10"

TRAIN_START = "2026-04-01"
TRAIN_END = "2026-04-20"

STATION_FILE = (
    ROOT
    / "02_RESULTS"
    / "DAY2_A_FIXED"
    / "day2_selected_candidates.csv"
)

FLOW_FILE = (
    ROOT
    / "02_RESULTS"
    / "DAY2_ID_FIXED"
    / "station_flow_summary.csv"
)

OUTPUT_DIR = (
    ROOT
    / "02_RESULTS"
    / "DAY4_HOURLY_PREP"
)


# ==================================================
# 2. 主程序
# ==================================================

def main():

    # ----------------------------------------------
    # 2.1 检查输入文件
    # ----------------------------------------------

    if not STATION_FILE.is_file():
        raise FileNotFoundError(STATION_FILE)

    if not FLOW_FILE.is_file():
        raise FileNotFoundError(FLOW_FILE)

    stations = pd.read_csv(
        STATION_FILE,
        dtype={
            "station_id": str,
            "candidate_seed": str
        }
    )

    stations = stations.loc[
        stations["candidate_seed"] == SEED
    ].copy()

    stations["proxy_capacity"] = pd.to_numeric(
        stations["proxy_capacity"],
        errors="raise"
    )

    if len(stations) != 25:
        raise ValueError(
            "主实验区域不是25个站点"
        )

    if stations["station_id"].duplicated().any():
        raise ValueError(
            "主实验区域存在重复站点"
        )

    if not stations["proxy_capacity"].gt(0).all():
        raise ValueError(
            "存在无效容量"
        )

    station_ids = stations[
        "station_id"
    ].tolist()

    # ----------------------------------------------
    # 2.2 训练工作日
    # ----------------------------------------------

    dates = pd.date_range(
        TRAIN_START,
        TRAIN_END,
        freq="D"
    )

    train_dates = [
        date.strftime("%Y-%m-%d")
        for date in dates
        if date.weekday() < 5
    ]

    n_days = len(train_dates)

    if n_days == 0:
        raise ValueError("没有训练工作日")

    date_set = set(train_dates)
    station_set = set(station_ids)

    # ----------------------------------------------
    # 2.3 读取小时借还数据
    # ----------------------------------------------

    print(
        "[1/3] 读取修复后的训练期小时借还事件..."
    )

    chunks = []

    reader = pd.read_csv(
        FLOW_FILE,
        dtype={
            "station_id": str,
            "date": str,
            "hour": "int16",
            "type": str,
            "count": "int64"
        },
        chunksize=250000
    )

    for chunk in reader:

        part = chunk.loc[
            chunk["station_id"].isin(station_set)
            & chunk["date"].isin(date_set),
            [
                "station_id",
                "date",
                "hour",
                "type",
                "count"
            ]
        ]

        if part.empty:
            continue

        if (part["count"] < 0).any():
            raise ValueError("存在负数事件计数")

        if not part["hour"].between(0, 23).all():
            raise ValueError("存在非法小时")

        if not part["type"].isin(
            ["start", "end"]
        ).all():
            raise ValueError("存在非法事件类型")

        chunks.append(part)

    if not chunks:
        raise ValueError(
            "训练期没有找到站点流量"
        )

    raw = pd.concat(
        chunks,
        ignore_index=True
    )

    # ----------------------------------------------
    # 2.4 统一每站、每天、每小时借还事件
    # ----------------------------------------------

    grouped = raw.groupby(
        [
            "station_id",
            "date",
            "hour",
            "type"
        ],
        as_index=False
    )["count"].sum()

    wide = grouped.pivot_table(
        index=[
            "station_id",
            "date",
            "hour"
        ],
        columns="type",
        values="count",
        aggfunc="sum",
        fill_value=0
    )

    for event_type in ["start", "end"]:
        if event_type not in wide.columns:
            wide[event_type] = 0

    # 补齐没有事件的小时
    complete_index = pd.MultiIndex.from_product(
        [
            station_ids,
            train_dates,
            range(24)
        ],
        names=[
            "station_id",
            "date",
            "hour"
        ]
    )

    daily_hourly = (
        wide[["start", "end"]]
        .reindex(
            complete_index,
            fill_value=0
        )
        .reset_index()
    )

    daily_hourly.columns.name = None

    daily_hourly = daily_hourly.rename(
        columns={
            "start": "observed_departures",
            "end": "observed_arrivals"
        }
    )

    daily_hourly["net_inflow"] = (
        daily_hourly["observed_arrivals"]
        - daily_hourly["observed_departures"]
    )

    # ----------------------------------------------
    # 2.5 建立24小时平均流量
    # ----------------------------------------------

    print(
        "[2/3] 生成每站24小时训练工作日平均流量..."
    )

    profiles = daily_hourly.groupby(
        [
            "station_id",
            "hour"
        ],
        as_index=False
    )[
        [
            "observed_departures",
            "observed_arrivals",
            "net_inflow"
        ]
    ].sum()

    profiles = profiles.rename(
        columns={
            "observed_departures": "avg_departures",
            "observed_arrivals": "avg_arrivals",
            "net_inflow": "avg_net_inflow"
        }
    )

    for column in [
        "avg_departures",
        "avg_arrivals",
        "avg_net_inflow"
    ]:

        profiles[column] /= n_days

    profiles = profiles.merge(
        stations[
            [
                "station_id",
                "station_name",
                "proxy_capacity"
            ]
        ],
        on="station_id",
        validate="many_to_one"
    )

    profiles["avg_net_as_capacity_ratio"] = (
        profiles["avg_net_inflow"]
        / profiles["proxy_capacity"]
    )

    profiles = profiles.sort_values(
        [
            "hour",
            "station_id"
        ]
    ).reset_index(drop=True)

    # ----------------------------------------------
    # 2.6 区域整体小时流量
    # ----------------------------------------------

    area = profiles.groupby(
        "hour",
        as_index=False
    )[
        [
            "avg_departures",
            "avg_arrivals",
            "avg_net_inflow"
        ]
    ].sum()

    area["cumulative_avg_net_inflow"] = (
        area["avg_net_inflow"].cumsum()
    )

    # ----------------------------------------------
    # 2.7 数据验收
    # ----------------------------------------------

    original_total = int(
        raw["count"].sum()
    )

    processed_total = int(
        daily_hourly[
            "observed_departures"
        ].sum()
        + daily_hourly[
            "observed_arrivals"
        ].sum()
    )

    if original_total != processed_total:
        raise AssertionError(
            "处理前后骑行事件数不一致"
        )

    assert len(profiles) == 25 * 24

    assert len(daily_hourly) == (
        25 * n_days * 24
    )

    if not np.allclose(
        profiles["avg_net_inflow"],
        (
            profiles["avg_arrivals"]
            - profiles["avg_departures"]
        )
    ):
        raise AssertionError(
            "小时净流量计算不一致"
        )

    # ----------------------------------------------
    # 2.8 保存结果
    # ----------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    daily_hourly.to_csv(
        OUTPUT_DIR / "train_station_day_hour.csv",
        index=False,
        encoding="utf-8-sig"
    )

    profiles.to_csv(
        OUTPUT_DIR / "train_hourly_station_profiles.csv",
        index=False,
        encoding="utf-8-sig"
    )

    area.to_csv(
        OUTPUT_DIR / "train_hourly_area_profiles.csv",
        index=False,
        encoding="utf-8-sig"
    )

    audit = {
        "seed": SEED,
        "station_count": len(station_ids),
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "train_weekdays": n_days,
        "proxy_capacity_total": float(
            stations["proxy_capacity"].sum()
        ),
        "observed_station_events": original_total,
        "event_conservation_pass": True,
        "limitations": [
            "历史计数仅包含已完成骑行。",
            "小时借还汇总不包含订单级OD对应关系。",
            "未得到真实未满足需求。",
            "容量来自当前GBFS快照。",
            "平均净流量不能直接当作真实库存。"
        ]
    }

    with open(
        OUTPUT_DIR / "day4_data_audit.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            audit,
            file,
            ensure_ascii=False,
            indent=2
        )

    # ----------------------------------------------
    # 2.9 终端摘要
    # ----------------------------------------------

    print(
        "[3/3] 输入表和审计报告已生成。"
    )

    print(
        f"PASS：站点={len(station_ids)}, "
        f"训练工作日={n_days}, "
        f"代理容量={stations['proxy_capacity'].sum():.0f}"
    )

    print(
        f"训练期站点借还事件总数："
        f"{original_total:,}"
    )

    print()
    print(
        "时间    平均借出    平均归还    平均净流入"
    )

    for row in area.loc[
        area["hour"].between(15, 20)
    ].itertuples(index=False):

        print(
            f"{int(row.hour):02d}:00  "
            f"{row.avg_departures:9.2f}  "
            f"{row.avg_arrivals:9.2f}  "
            f"{row.avg_net_inflow:11.2f}"
        )

    print()
    print("输出目录：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
