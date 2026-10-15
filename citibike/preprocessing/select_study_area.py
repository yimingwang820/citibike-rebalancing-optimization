
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 2-A: Citi Bike 候选站点区域比较。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def haversine_km(lat1, lon1, lat2, lon2):
    """计算经纬度之间的球面距离，单位 km。"""
    r = 6371.0088
    a1, a2 = np.radians(lat1), np.radians(lat2)
    da = a2 - a1
    dl = np.radians(lon2 - lon1)
    a = (
        np.sin(da / 2) ** 2
        + np.cos(a1) * np.cos(a2) * np.sin(dl / 2) ** 2
    )
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_registry(registry_file, metadata_file):
    """读取站点表，并按 GBFS short_name 重新匹配当前容量。"""
    reg = pd.read_csv(
        registry_file, dtype={"station_id": str}
    )

    required = {
        "station_id", "station_name", "latitude", "longitude"
    }
    missing = required - set(reg.columns)

    if missing:
        raise ValueError(f"站点表缺少字段: {sorted(missing)}")

    reg["station_id"] = reg["station_id"].str.strip()

    if reg["station_id"].duplicated().any():
        raise ValueError("站点表存在重复 station_id")

    for field in ["latitude", "longitude"]:
        reg[field] = pd.to_numeric(
            reg[field], errors="coerce"
        )

    raw = json.loads(
        metadata_file.read_text(encoding="utf-8")
    )

    records = []

    for item in raw.get("data", {}).get("stations", []):
        sid = str(item.get("short_name") or "").strip()

        if sid:
            records.append({
                "station_id": sid,
                "proxy_capacity": pd.to_numeric(
                    item.get("capacity"), errors="coerce"
                ),
                "gbfs_station_id": str(
                    item.get("station_id") or ""
                ),
            })

    if not records:
        raise ValueError("GBFS JSON 中没有有效 short_name")

    meta = pd.DataFrame(records)

    if meta["station_id"].duplicated().any():
        raise ValueError("GBFS short_name 存在重复")

    # 不信任旧版站点表中的容量匹配结果。
    reg = reg.drop(
        columns=[
            "capacity", "capacity_source",
            "gbfs_station_id"
        ],
        errors="ignore"
    )

    reg = reg.merge(
        meta,
        on="station_id",
        how="left",
        validate="one_to_one"
    )

    reg["proxy_capacity"] = pd.to_numeric(
        reg["proxy_capacity"], errors="coerce"
    )

    reg["has_positive_capacity"] = (
        reg["proxy_capacity"] > 0
    )

    reg["capacity_source"] = np.where(
        reg["has_positive_capacity"],
        "CURRENT_GBFS_POSITIVE_NOT_HISTORICAL",
        "UNAVAILABLE_OR_NONPOSITIVE"
    )

    return reg


def read_train_activity(flow_file, start, end):
    """按训练窗口计算活动量和活跃天数。"""
    parts = []

    reader = pd.read_csv(
        flow_file,
        usecols=[
            "station_id", "date", "hour", "type", "count"
        ],
        dtype={
            "station_id": str,
            "date": str,
            "type": str,
            "hour": "int16",
            "count": "int64",
        },
        chunksize=250000
    )

    for chunk in reader:
        chunk = chunk.loc[
            chunk["date"].between(start, end)
            & chunk["type"].isin(["start", "end"])
        ]

        if chunk.empty:
            continue

        if (
            (chunk["count"] < 0).any()
            or (~chunk["hour"].between(0, 23)).any()
        ):
            raise ValueError("小时流量中存在非法数据")

        part = chunk.groupby(
            ["station_id", "date"], as_index=False
        )["count"].sum()

        parts.append(part)

    if not parts:
        raise ValueError("训练窗口内没有流量数据")

    daily = pd.concat(
        parts, ignore_index=True
    ).groupby(
        ["station_id", "date"], as_index=False
    )["count"].sum()

    activity = daily.groupby("station_id").agg(
        train_activity=("count", "sum"),
        train_active_days=("date", "nunique")
    ).reset_index()

    return activity


def build_candidates(reg, seeds, k, min_days):
    """为每个固定种子选择最近的 k 个合格站点。"""
    eligible = reg.loc[
        reg["has_positive_capacity"]
        & reg["latitude"].between(-90, 90)
        & reg["longitude"].between(-180, 180)
        & reg["train_active_days"].ge(min_days)
        & reg["train_activity"].gt(0)
    ].copy()

    if len(eligible) < k:
        raise ValueError(
            f"合格站点只有 {len(eligible)} 个，不足 {k} 个"
        )

    candidates = {}

    for seed in seeds:
        seedrow = eligible.loc[
            eligible["station_id"] == seed
        ]

        if seedrow.empty:
            raise ValueError(
                f"种子站点 {seed} 不符合筛选条件"
            )

        s = seedrow.iloc[0]

        table = eligible.copy()

        table["distance_to_seed_km"] = haversine_km(
            s["latitude"],
            s["longitude"],
            table["latitude"].to_numpy(),
            table["longitude"].to_numpy()
        )

        table = table.sort_values(
            [
                "distance_to_seed_km",
                "train_activity",
                "station_id"
            ],
            ascending=[True, False, True]
        ).head(k).copy()

        table["candidate_seed"] = seed

        assert seed in set(table["station_id"])
        assert table["has_positive_capacity"].all()
        assert len(table) == k

        candidates[seed] = table

    return candidates, eligible


def read_selected_hourly(flow_file, ids, start, end):
    """只读取入选站点在训练窗口中的小时事件。"""
    pieces = []

    reader = pd.read_csv(
        flow_file,
        usecols=[
            "station_id", "date", "hour", "type", "count"
        ],
        dtype={
            "station_id": str,
            "date": str,
            "type": str,
            "hour": "int16",
            "count": "int64",
        },
        chunksize=250000
    )

    for chunk in reader:
        chunk = chunk.loc[
            chunk["date"].between(start, end)
            & chunk["station_id"].isin(ids)
            & chunk["type"].isin(["start", "end"])
        ]

        if not chunk.empty:
            pieces.append(chunk)

    if not pieces:
        raise ValueError("入选站点没有训练期流量")

    observed = pd.concat(
        pieces, ignore_index=True
    ).groupby(
        ["station_id", "date", "hour", "type"],
        as_index=False
    )["count"].sum()

    wide = observed.pivot_table(
        index=["station_id", "date", "hour"],
        columns="type",
        values="count",
        aggfunc="sum",
        fill_value=0
    )

    for col in ["start", "end"]:
        if col not in wide.columns:
            wide[col] = 0

    dates = pd.date_range(
        start, end, freq="D"
    ).strftime("%Y-%m-%d").tolist()

    # 补齐每天 24 小时，没有事件的小时填 0。
    index = pd.MultiIndex.from_product(
        [sorted(ids), dates, list(range(24))],
        names=["station_id", "date", "hour"]
    )

    wide = wide[["start", "end"]].reindex(
        index, fill_value=0
    ).reset_index()

    wide.columns.name = None

    wide["start"] = wide["start"].astype("int64")
    wide["end"] = wide["end"].astype("int64")

    wide["net"] = wide["end"] - wide["start"]

    wide["weekday"] = (
        pd.to_datetime(wide["date"]).dt.dayofweek < 5
    )

    return wide


def station_pressure(hourly, station_ids, cap_table,
                     start, end):
    """计算站点的累计净流量波动与早晚高峰指标。"""
    sub = hourly.loc[
        hourly["station_id"].isin(station_ids)
    ].copy()

    sub = sub.sort_values(
        ["station_id", "date", "hour"]
    )

    sub["cumulative_net"] = sub.groupby(
        ["station_id", "date"]
    )["net"].cumsum()

    bounds = sub.groupby(
        ["station_id", "date"]
    )["cumulative_net"].agg(["min", "max"])

    # 加入每日开始时的累计净流量 0。
    bounds["daily_net_range"] = (
        bounds["max"].clip(lower=0)
        - bounds["min"].clip(upper=0)
    )

    ranges = bounds.groupby(
        level="station_id"
    )["daily_net_range"].agg(
        mean_daily_net_range="mean",
        median_daily_net_range="median"
    ).reset_index()

    totals = sub.groupby(
        "station_id", as_index=False
    ).agg(
        train_departures=("start", "sum"),
        train_arrivals=("end", "sum")
    )

    week = sub.loc[sub["weekday"]]

    n_weekdays = sum(
        pd.date_range(start, end).dayofweek < 5
    )

    def average_net_at_hours(start_h, end_h):
        part = week.loc[
            week["hour"].between(start_h, end_h)
        ]

        result = (
            part.groupby("station_id")["net"].sum()
            / n_weekdays
        )

        return result.rename(
            f"weekday_net_{start_h:02d}_{end_h:02d}"
        ).reset_index()

    result = cap_table[
        [
            "station_id",
            "station_name",
            "latitude",
            "longitude",
            "proxy_capacity",
            "capacity_source",
            "train_activity",
            "train_active_days",
            "distance_to_seed_km",
            "candidate_seed",
        ]
    ].copy()

    result = result.merge(
        ranges,
        on="station_id",
        validate="one_to_one"
    )

    result = result.merge(
        totals,
        on="station_id",
        validate="one_to_one"
    )

    result = result.merge(
        average_net_at_hours(7, 9),
        on="station_id",
        validate="one_to_one"
    )

    result = result.merge(
        average_net_at_hours(16, 18),
        on="station_id",
        validate="one_to_one"
    )

    result["pressure_ratio"] = (
        result["mean_daily_net_range"]
        / result["proxy_capacity"]
    )

    result["train_net_inflow"] = (
        result["train_arrivals"]
        - result["train_departures"]
    )

    return result


def boundary_from_fullmonth_od(od_file, candidates):
    """计算全月 OD 边界诊断；不得用于训练期选站。"""
    stats = {
        seed: {
            "internal": 0,
            "incoming": 0,
            "outgoing": 0
        }
        for seed in candidates
    }

    nodes = {
        seed: set(tab["station_id"])
        for seed, tab in candidates.items()
    }

    reader = pd.read_csv(
        od_file,
        dtype={
            "start_station_id": str,
            "end_station_id": str,
            "trips": "int64"
        },
        chunksize=250000
    )

    for chunk in reader:
        for seed, members in nodes.items():
            a = chunk["start_station_id"].isin(members)
            b = chunk["end_station_id"].isin(members)
            n = chunk["trips"]

            stats[seed]["internal"] += int(
                n[a & b].sum()
            )

            stats[seed]["incoming"] += int(
                n[~a & b].sum()
            )

            stats[seed]["outgoing"] += int(
                n[a & ~b].sum()
            )

    return stats


def spatial_stats(table):
    """计算站点间平均距离及最大距离。"""
    lat = table["latitude"].to_numpy()
    lon = table["longitude"].to_numpy()

    d = haversine_km(
        lat[:, None],
        lon[:, None],
        lat[None, :],
        lon[None, :]
    )

    upper = d[np.triu_indices(len(table), k=1)]

    return float(upper.mean()), float(upper.max())


def save_charts(candidates, profiles, eligible, out_dir):
    """生成空间分布和容量归一化失衡对比图。"""
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(
        eligible["longitude"],
        eligible["latitude"],
        s=6,
        alpha=0.12,
        label="Eligible stations"
    )

    for seed, tab in candidates.items():
        ax.scatter(
            tab["longitude"],
            tab["latitude"],
            s=28,
            alpha=0.78,
            label=f"Candidate {seed}"
        )

        row = tab.loc[
            tab["station_id"] == seed
        ].iloc[0]

        ax.annotate(
            seed,
            (row["longitude"], row["latitude"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )

    ax.set(
        xlabel="Longitude",
        ylabel="Latitude",
        title="Rebuilt 25-station candidate networks"
    )
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        out_dir / "01_candidate_locations.png",
        dpi=170
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))

    for seed, part in profiles.groupby("candidate_seed"):
        data = part.sort_values(
            "pressure_ratio", ascending=False
        )["pressure_ratio"].to_numpy()

        ax.plot(
            np.arange(1, len(data) + 1),
            data,
            marker="o",
            markersize=3,
            label=seed
        )

    ax.set(
        xlabel="Station rank",
        ylabel="Mean daily net range / proxy capacity",
        title="Training-period station pressure proxy"
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        out_dir / "02_pressure_ratio_profiles.png",
        dpi=170
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Day 2-A Citi Bike 候选站点分析"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("02_RESULTS")
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("00_DATA/station_information.json")
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("02_RESULTS/DAY2_A")
    )

    parser.add_argument(
        "--start",
        default="2026-04-01"
    )

    parser.add_argument(
        "--train-end",
        default="2026-04-20"
    )

    parser.add_argument(
        "--stations",
        type=int,
        default=25
    )

    parser.add_argument(
        "--min-days",
        type=int,
        default=12
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["6140.05", "5980.10"]
    )

    args = parser.parse_args()

    if args.stations < 2 or args.min_days < 1:
        parser.error("站点数至少为 2，活跃天数至少为 1")

    if args.start > args.train_end:
        parser.error("训练开始日期不能晚于结束日期")

    files = {
        "registry": args.input_dir / "station_registry.csv",
        "flow": args.input_dir / "station_flow_summary.csv",
        "od": args.input_dir / "station_od_counts.csv",
        "clusters": args.input_dir / "candidate_clusters.csv",
        "metadata": args.metadata,
    }

    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"缺少 {name} 文件: {path.resolve()}"
            )

    args.output_dir.mkdir(
        parents=True, exist_ok=True
    )

    # 第一阶段：匹配容量。
    print("[1/5] 核对站点与 GBFS 容量...")

    reg = load_registry(
        files["registry"],
        files["metadata"]
    )

    print(
        f"历史站点: {len(reg)}；"
        f"当前元数据匹配: "
        f"{reg['gbfs_station_id'].notna().sum()}；"
        f"正容量: {reg['has_positive_capacity'].sum()}"
    )

    # 第二阶段：训练期活动统计。
    print("[2/5] 统计训练期站点活动...")

    activity = read_train_activity(
        files["flow"],
        args.start,
        args.train_end
    )

    reg = reg.merge(
        activity,
        how="left",
        on="station_id",
        validate="one_to_one"
    )

    reg[
        ["train_activity", "train_active_days"]
    ] = reg[
        ["train_activity", "train_active_days"]
    ].fillna(0).astype("int64")

    # 第三阶段：重新构建候选网络。
    candidates, eligible = build_candidates(
        reg,
        args.seeds,
        args.stations,
        args.min_days
    )

    ids = set().union(*[
        set(tab["station_id"])
        for tab in candidates.values()
    ])

    print(
        f"[3/5] 已构建候选网络，"
        f"共涉及 {len(ids)} 个不同站点..."
    )

    hourly = read_selected_hourly(
        files["flow"],
        ids,
        args.start,
        args.train_end
    )

    profiles = pd.concat(
        [
            station_pressure(
                hourly,
                set(tab["station_id"]),
                tab,
                args.start,
                args.train_end
            )
            for tab in candidates.values()
        ],
        ignore_index=True
    )

    # 第四阶段：全月边界诊断。
    print("[4/5] 计算全月 OD 边界诊断...")

    boundary = boundary_from_fullmonth_od(
        files["od"], candidates
    )

    original = pd.read_csv(
        files["clusters"],
        dtype={"seed_station_id": str}
    )

    original_lookup = {
        row["seed_station_id"]:
        set(str(row["station_ids"]).split(";"))
        for _, row in original.iterrows()
    }

    rows = []

    for seed, tab in candidates.items():
        part = profiles.loc[
            profiles["candidate_seed"] == seed
        ]

        mean_pairwise, max_pairwise = spatial_stats(tab)

        b = boundary[seed]

        related = (
            b["internal"]
            + b["incoming"]
            + b["outgoing"]
        )

        old_set = original_lookup.get(seed, set())
        new_set = set(tab["station_id"])

        rows.append({
            "candidate_seed": seed,
            "station_count": len(tab),
            "positive_capacity_count": int(
                tab["proxy_capacity"].gt(0).sum()
            ),
            "max_radius_km": round(
                float(tab["distance_to_seed_km"].max()), 4
            ),
            "mean_pairwise_distance_km": round(
                mean_pairwise, 4
            ),
            "max_pairwise_distance_km": round(
                max_pairwise, 4
            ),
            "train_activity_events": int(
                part["train_activity"].sum()
            ),
            "mean_train_active_days": round(
                float(part["train_active_days"].mean()), 2
            ),
            "mean_daily_net_range": round(
                float(part["mean_daily_net_range"].mean()), 3
            ),
            "mean_pressure_ratio": round(
                float(part["pressure_ratio"].mean()), 4
            ),
            "median_pressure_ratio": round(
                float(part["pressure_ratio"].median()), 4
            ),
            "stations_pressure_gt_1": int(
                part["pressure_ratio"].gt(1).sum()
            ),
            "weekday_morning_negative_stations": int(
                part["weekday_net_07_09"].lt(0).sum()
            ),
            "weekday_evening_negative_stations": int(
                part["weekday_net_16_18"].lt(0).sum()
            ),
            "weekday_morning_avg_net_total": round(
                float(part["weekday_net_07_09"].sum()), 2
            ),
            "weekday_evening_avg_net_total": round(
                float(part["weekday_net_16_18"].sum()), 2
            ),
            "internal_trips_full_month_DIAGNOSTIC":
                b["internal"],
            "incoming_trips_full_month_DIAGNOSTIC":
                b["incoming"],
            "outgoing_trips_full_month_DIAGNOSTIC":
                b["outgoing"],
            "cross_boundary_share_full_month_DIAGNOSTIC":
                round(
                    (
                        b["incoming"] + b["outgoing"]
                    ) / related,
                    4
                ) if related else np.nan,
            "overlap_with_original_cluster": len(
                new_set & old_set
            ),
        })

    comparison = pd.DataFrame(rows).sort_values(
        "candidate_seed"
    )

    members = pd.concat(
        candidates.values(),
        ignore_index=True
    )

    members = members[
        [
            "candidate_seed",
            "station_id",
            "station_name",
            "latitude",
            "longitude",
            "proxy_capacity",
            "capacity_source",
            "train_activity",
            "train_active_days",
            "distance_to_seed_km",
        ]
    ]

    profiles = profiles.sort_values(
        ["candidate_seed", "pressure_ratio"],
        ascending=[True, False]
    )

    # 第五阶段：保存输出。
    comparison.to_csv(
        args.output_dir / "day2_candidate_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    profiles.to_csv(
        args.output_dir / "day2_station_profiles.csv",
        index=False,
        encoding="utf-8-sig"
    )

    members.to_csv(
        args.output_dir / "day2_selected_candidates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    save_charts(
        candidates,
        profiles,
        eligible,
        args.output_dir
    )

    intersection = set.intersection(
        *[
            set(tab["station_id"])
            for tab in candidates.values()
        ]
    )

    lines = [
        "# Day 2-A 候选站点网络诊断报告",
        "",
        f"- 训练窗口：{args.start} 至 {args.train_end}",
        f"- 每个候选网络站点数：{args.stations}",
        f"- 最低活跃天数：{args.min_days}",
        f"- 候选网络共同站点数：{len(intersection)}",
        "- 容量来自当前 GBFS 快照，不是历史容量。",
        "- 历史订单仅包含已完成骑行。",
        "- 全月 OD 指标仅用于诊断，不参与选站。",
        "- 压力比不等于真实缺车率。",
        "- 周一至周五视为工作日，未额外排除节假日。",
        "- 目前尚未确定最终研究区域。",
        "",
        "## 训练期候选比较",
        "",
        "| 种子 | 半径 km | 训练事件 | 平均压力比 | 早高峰净流出站点 | 晚高峰净流出站点 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, r in comparison.iterrows():
        lines.append(
            f"| {r.candidate_seed} "
            f"| {r.max_radius_km:.3f} "
            f"| {int(r.train_activity_events):,} "
            f"| {r.mean_pressure_ratio:.3f} "
            f"| {int(r.weekday_morning_negative_stations)} "
            f"| {int(r.weekday_evening_negative_stations)} |"
        )

    lines.extend([
        "",
        "## 下一步",
        "",
        "1. 检查两个候选网络的空间范围。",
        "2. 比较早晚高峰净流量与失衡特征。",
        "3. 人工确认最终 25 个站点。",
        "4. 再建立初始库存情景和 Gurobi 模型。",
        "",
    ])

    (
        args.output_dir / "day2_selection_report.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("[5/5] 结果生成完成！")

    for _, r in comparison.iterrows():
        print(
            f"  种子 {r.candidate_seed}: "
            f"半径={r.max_radius_km:.3f}km, "
            f"训练活动={int(r.train_activity_events):,}, "
            f"压力比={r.mean_pressure_ratio:.3f}, "
            f"正容量={int(r.positive_capacity_count)}"
            f"/{args.stations}"
        )

    print(f"  共同站点: {len(intersection)}")
    print("  输出目录:", args.output_dir.resolve())
    print("  注意：尚未自动确定最终研究区域。")


if __name__ == "__main__":
    main()
