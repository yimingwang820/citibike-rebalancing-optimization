
# -*- coding: utf-8 -*-
"""
Day 5-A：Citi Bike 独立测试日订单提取

功能：
1. 从2026年4月原始ZIP读取订单
2. 使用修复后的站点ID映射
3. 保留与主区域25站相关的完整订单
4. 保留跨区域和跨日期的起终点对应关系
5. 与Day1修复后的小时汇总进行对账

不需要Gurobi。
直接运行，无需命令行参数。
"""

from pathlib import Path
from zipfile import ZipFile
import json

import pandas as pd


# ==================================================
# 1. 路径及参数
# ==================================================

from citibike.paths import ROOT

TARGET_DATE = "2026-04-21"
SEED = "5980.10"

ZIP_FILE = (
    ROOT / "00_DATA"
    / "202604-citibike-tripdata.zip"
)

STATION_FILE = (
    ROOT / "02_RESULTS"
    / "DAY2_A_FIXED"
    / "day2_selected_candidates.csv"
)

MAPPING_FILE = (
    ROOT / "02_RESULTS"
    / "DAY2_ID_FIXED"
    / "station_id_mapping.csv"
)

FLOW_FILE = (
    ROOT / "02_RESULTS"
    / "DAY2_ID_FIXED"
    / "station_flow_summary.csv"
)

OUTPUT_DIR = (
    ROOT / "02_RESULTS"
    / "DAY5_EVENT_DATA"
)

COLUMNS = [
    "ride_id",
    "started_at",
    "ended_at",
    "start_station_id",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng"
]


# ==================================================
# 2. 主程序
# ==================================================

def main():

    for path in [
        ZIP_FILE,
        STATION_FILE,
        MAPPING_FILE,
        FLOW_FILE
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"缺少输入文件：{path}"
            )

    # ----------------------------------------------
    # 2.1 冻结的主实验站点
    # ----------------------------------------------

    stations = pd.read_csv(
        STATION_FILE,
        dtype={
            "station_id": str,
            "candidate_seed": str
        }
    )

    selected = set(
        stations.loc[
            stations["candidate_seed"] == SEED,
            "station_id"
        ]
    )

    if len(selected) != 25:
        raise ValueError(
            "主实验区域不是25个站点"
        )

    # ----------------------------------------------
    # 2.2 原始ID到规范ID的映射
    # ----------------------------------------------

    mapping = pd.read_csv(
        MAPPING_FILE,
        dtype={
            "raw_station_id": str,
            "canonical_station_id": str
        }
    )

    if mapping["raw_station_id"].duplicated().any():
        raise ValueError(
            "ID映射表存在重复原始ID"
        )

    aliases = dict(zip(
        mapping["raw_station_id"],
        mapping["canonical_station_id"]
    ))

    # ----------------------------------------------
    # 2.3 流式读取原始ZIP
    # ----------------------------------------------

    print(
        "[1/4] 从历史ZIP读取测试日相关订单..."
    )

    parts = []
    seen_ids = set()

    scanned = 0
    candidates = 0
    invalid = 0
    duplicates = 0

    with ZipFile(ZIP_FILE) as archive:

        csv_files = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
            and not name.startswith("__MACOSX/")
        )

        if not csv_files:
            raise ValueError(
                "ZIP中没有CSV文件"
            )

        print(
            f"ZIP包含 {len(csv_files)} 个CSV文件"
        )

        for filename in csv_files:

            with archive.open(filename) as file:

                reader = pd.read_csv(
                    file,
                    usecols=COLUMNS,
                    chunksize=100000,
                    low_memory=False,
                    dtype={
                        "ride_id": str,
                        "start_station_id": str,
                        "end_station_id": str
                    }
                )

                for chunk in reader:

                    scanned += len(chunk)

                    raw_start = (
                        chunk["start_station_id"]
                        .fillna("")
                        .str.strip()
                    )

                    raw_end = (
                        chunk["end_station_id"]
                        .fillna("")
                        .str.strip()
                    )

                    canonical_start = (
                        raw_start.map(aliases)
                        .fillna(raw_start)
                    )

                    canonical_end = (
                        raw_end.map(aliases)
                        .fillna(raw_end)
                    )

                    # 至少一个端点在主区域内
                    inside = (
                        canonical_start.isin(selected)
                        | canonical_end.isin(selected)
                    )

                    # 借出或归还发生在测试日
                    day_match = (
                        chunk["started_at"]
                        .astype("string")
                        .str[:10]
                        .eq(TARGET_DATE)
                        |
                        chunk["ended_at"]
                        .astype("string")
                        .str[:10]
                        .eq(TARGET_DATE)
                    )

                    chosen = inside & day_match

                    if not chosen.any():
                        continue

                    part = chunk.loc[chosen].copy()
                    candidates += len(part)

                    # 保留原始ID作为追溯证据
                    part["raw_start_station_id"] = (
                        raw_start.loc[chosen].to_numpy()
                    )

                    part["raw_end_station_id"] = (
                        raw_end.loc[chosen].to_numpy()
                    )

                    # 规范化后的站点ID
                    part["start_station_id"] = (
                        canonical_start.loc[
                            chosen
                        ].to_numpy()
                    )

                    part["end_station_id"] = (
                        canonical_end.loc[
                            chosen
                        ].to_numpy()
                    )

                    # ----------------------------------
                    # 数据质量检查
                    # ----------------------------------

                    start = pd.to_datetime(
                        part["started_at"],
                        errors="coerce"
                    )

                    end = pd.to_datetime(
                        part["ended_at"],
                        errors="coerce"
                    )

                    duration = (
                        end - start
                    ).dt.total_seconds()

                    valid = (
                        start.notna()
                        & end.notna()
                        & duration.gt(0)
                        & duration.le(86400)
                        & start.dt.strftime(
                            "%Y-%m"
                        ).eq("2026-04")
                    )

                    ride_ids = (
                        part["ride_id"]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                    )

                    valid &= ride_ids.ne("")

                    valid &= (
                        part["start_station_id"].ne("")
                        & part["end_station_id"].ne("")
                    )

                    for column in [
                        "start_lat",
                        "start_lng",
                        "end_lat",
                        "end_lng"
                    ]:
                        part[column] = pd.to_numeric(
                            part[column],
                            errors="coerce"
                        )

                    valid &= (
                        part["start_lat"].between(-90, 90)
                        & part["end_lat"].between(-90, 90)
                        & part["start_lng"].between(-180, 180)
                        & part["end_lng"].between(-180, 180)
                    )

                    # 仅在已筛出的相关订单内检查重复ID
                    duplicate = (
                        ride_ids.duplicated(
                            keep="first"
                        )
                        | ride_ids.isin(seen_ids)
                    )

                    duplicates += int(
                        (duplicate & valid).sum()
                    )

                    seen_ids.update(
                        ride_ids[
                            ride_ids.ne("")
                        ].tolist()
                    )

                    valid &= ~duplicate

                    invalid += int(
                        (~valid).sum()
                    )

                    part = part.loc[valid].copy()

                    if part.empty:
                        continue

                    # ----------------------------------
                    # 保存实际开始和结束日期、小时
                    # ----------------------------------

                    part["start_date"] = (
                        start.loc[valid]
                        .dt.strftime("%Y-%m-%d")
                        .to_numpy()
                    )

                    part["end_date"] = (
                        end.loc[valid]
                        .dt.strftime("%Y-%m-%d")
                        .to_numpy()
                    )

                    part["start_hour"] = (
                        start.loc[valid]
                        .dt.hour.to_numpy()
                    )

                    part["end_hour"] = (
                        end.loc[valid]
                        .dt.hour.to_numpy()
                    )

                    s_in = (
                        part["start_station_id"]
                        .isin(selected)
                    )

                    e_in = (
                        part["end_station_id"]
                        .isin(selected)
                    )

                    # 区分区域内部、流出和流入订单
                    part["flow_class"] = ""

                    part.loc[
                        s_in & e_in,
                        "flow_class"
                    ] = "internal"

                    part.loc[
                        s_in & ~e_in,
                        "flow_class"
                    ] = "outbound"

                    part.loc[
                        ~s_in & e_in,
                        "flow_class"
                    ] = "inbound"

                    parts.append(part)

    if not parts:
        raise RuntimeError(
            "没有提取到相关订单，"
            "请检查ZIP和测试日期"
        )

    trips = pd.concat(
        parts,
        ignore_index=True
    )

    trips = trips.sort_values(
        ["started_at", "ride_id"]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    trips.to_csv(
        OUTPUT_DIR / "test_day_trip_pairs.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ==================================================
    # 3. 重建单日、逐站、逐小时事件计数
    # ==================================================

    print(
        "[2/4] 重建主区域逐小时借还事件..."
    )

    index = pd.MultiIndex.from_product(
        [
            sorted(selected),
            range(24)
        ],
        names=[
            "station_id",
            "hour"
        ]
    )

    day_departures = trips.loc[
        (trips["start_date"] == TARGET_DATE)
        & trips["start_station_id"].isin(selected)
    ]

    day_arrivals = trips.loc[
        (trips["end_date"] == TARGET_DATE)
        & trips["end_station_id"].isin(selected)
    ]

    actual = pd.DataFrame(index=index)

    actual["start"] = (
        day_departures.groupby(
            ["start_station_id", "start_hour"]
        )
        .size()
        .reindex(index, fill_value=0)
        .to_numpy()
    )

    actual["end"] = (
        day_arrivals.groupby(
            ["end_station_id", "end_hour"]
        )
        .size()
        .reindex(index, fill_value=0)
        .to_numpy()
    )

    # ==================================================
    # 4. 与修复后的Day1小时汇总对账
    # ==================================================

    print(
        "[3/4] 与修复后的小时汇总对账..."
    )

    flow = pd.read_csv(
        FLOW_FILE,
        dtype={
            "station_id": str,
            "date": str,
            "type": str
        }
    )

    flow = flow.loc[
        (flow["date"] == TARGET_DATE)
        & flow["station_id"].isin(selected)
    ]

    expected = flow.pivot_table(
        index=["station_id", "hour"],
        columns="type",
        values="count",
        aggfunc="sum",
        fill_value=0
    ).reindex(
        index,
        fill_value=0
    )

    for column in ["start", "end"]:
        if column not in expected.columns:
            expected[column] = 0

    check = actual.rename(
        columns={
            "start": "extracted_start",
            "end": "extracted_end"
        }
    ).copy()

    check["day1_start"] = (
        expected["start"].to_numpy()
    )

    check["day1_end"] = (
        expected["end"].to_numpy()
    )

    check["start_delta"] = (
        check["extracted_start"]
        - check["day1_start"]
    )

    check["end_delta"] = (
        check["extracted_end"]
        - check["day1_end"]
    )

    check.reset_index().to_csv(
        OUTPUT_DIR
        / "test_day_hourly_reconciliation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    mismatches = int(
        (
            (check["start_delta"] != 0)
            | (check["end_delta"] != 0)
        ).sum()
    )

    counts = trips["flow_class"].value_counts()

    internal = int(
        counts.get("internal", 0)
    )

    inbound = int(
        counts.get("inbound", 0)
    )

    outbound = int(
        counts.get("outbound", 0)
    )

    print(
        f"提取订单：{len(trips):,}"
    )

    print(
        f"内部订单：{internal:,}"
    )

    print(
        f"外部流入订单：{inbound:,}"
    )

    print(
        f"外部流出订单：{outbound:,}"
    )

    print(
        f"测试日区域借出事件："
        f"{len(day_departures):,}"
    )

    print(
        f"测试日区域归还事件："
        f"{len(day_arrivals):,}"
    )

    print(
        f"小时统计不一致的站点-小时数："
        f"{mismatches}"
    )

    # ==================================================
    # 5. 保存数据审计报告
    # ==================================================

    audit = {
        "target_date": TARGET_DATE,
        "station_seed": SEED,
        "selected_stations": len(selected),
        "raw_rows_scanned": scanned,
        "date_area_candidates": candidates,
        "retained_trip_pairs": len(trips),
        "invalid_or_duplicate_candidates": invalid,
        "duplicate_relevant_ride_ids": duplicates,
        "internal_trips": internal,
        "inbound_trips": inbound,
        "outbound_trips": outbound,
        "station_departures": len(day_departures),
        "station_arrivals": len(day_arrivals),
        "hourly_mismatch_rows": mismatches,
        "hourly_reconciliation_pass": (
            mismatches == 0
        ),
        "notes": [
            "包含起点或终点在主区域，"
            "且至少一个事件发生在测试日的订单。",
            "保留原始ID与规范化ID。",
            "保留订单真实开始、结束时间。",
            "允许订单跨日期和跨区域。",
            "仅在筛出的相关订单内检查重复ride_id，"
            "与Day1全量去重顺序不完全等价。",
            "本步骤只准备事件数据，不模拟服务拒绝。",
            "库存及容量仍需另行建模。"
        ]
    }

    (
        OUTPUT_DIR / "day5_extract_audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "[4/4] 数据保存完成"
    )

    if mismatches == 0:
        print(
            "PASS：订单级小时计数"
            "与已修复汇总完全一致"
        )
    else:
        print(
            "WARNING：小时计数存在差异，"
            "暂不能通过验收"
        )

    print(
        "输出目录：", OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
