#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Citi Bike 站点 ID 格式别名修复：保留原始文件，输出规范化副本及审计记录。

示例（在项目根目录）：
python -m citibike.preprocessing.normalize_station_ids --input-dir 02_RESULTS \
    --metadata 00_DATA/station_information.json --output-dir 02_RESULTS/DAY2_ID_FIXED
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd


ID_PATTERN = re.compile(r"^[0-9]{3,5}\.[0-9]{1,3}$")


def numeric_key(station_id: str) -> str | None:
    """只用于识别候选别名。不会把所有字符串简单转换为浮点数。"""
    sid = str(station_id).strip()
    if not ID_PATTERN.fullmatch(sid):
        return None
    return format(Decimal(sid).normalize(), "f")


def comparable_name(name: object) -> str:
    return " ".join(str(name or "").casefold().split())


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = radians(lat1), radians(lat2)
    dp = p2 - p1
    dl = radians(lon2 - lon1)
    h = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * 6371008.8 * asin(sqrt(min(1.0, max(0.0, h))))


def load_gbfs(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = {}
    by_key = {}
    for item in raw.get("data", {}).get("stations", []):
        sid = str(item.get("short_name") or "").strip()
        if not sid:
            continue
        if sid in records:
            raise ValueError(f"GBFS short_name 重复: {sid}")
        records[sid] = item
        key = numeric_key(sid)
        if key:
            by_key.setdefault(key, []).append(sid)
    if not records:
        raise ValueError("GBFS 缺少可用的 short_name")
    return records, by_key


def make_mapping(registry: pd.DataFrame, gbfs_by_key: dict, tolerance_m: float):
    """保守归并：编号数值等价 + 同站名 + 坐标接近；不满足则保持原 ID。"""
    registry = registry.copy()
    registry["id_numeric_key"] = registry["station_id"].map(numeric_key)
    mapping = {sid: sid for sid in registry["station_id"]}
    rows, reviews = [], []

    for key, group in registry.dropna(subset=["id_numeric_key"]).groupby("id_numeric_key"):
        if len(group) < 2:
            continue
        ids = sorted(group["station_id"].tolist())
        names = {comparable_name(v) for v in group["station_name"]}
        coordinates = list(zip(group["latitude"], group["longitude"]))
        max_dist = max(
            distance_m(float(a), float(b), float(c), float(d))
            for a, b in coordinates for c, d in coordinates
        )
        gbfs_ids = gbfs_by_key.get(key, [])
        reasons = []
        if len(names) != 1 or "" in names:
            reasons.append("历史站点名称不一致或为空")
        if not all(pd.notna(lat) and pd.notna(lon) for lat, lon in coordinates):
            reasons.append("坐标缺失")
        elif max_dist > tolerance_m:
            reasons.append(f"组内坐标最大差异 {max_dist:.2f}m 超过阈值")
        if len(gbfs_ids) > 1:
            reasons.append("当前 GBFS 存在多个等价编号")

        if reasons:
            reviews.append({
                "id_numeric_key": key,
                "raw_station_ids": ";".join(ids),
                "issue": ";".join(reasons),
                "max_distance_m": max_dist,
            })
            continue

        # 首选 GBFS 当前 short_name；若不存在，则使用历史编号中小数位数最多的写法。
        if len(gbfs_ids) == 1:
            canonical = gbfs_ids[0]
            selection = "CURRENT_GBFS_SHORT_NAME"
        else:
            canonical = sorted(
                ids,
                key=lambda s: (-len(s.split(".")[-1]), -int(group.loc[
                    group["station_id"] == s, "total_activity"
                ].iloc[0]), s),
            )[0]
            selection = "HISTORICAL_CONSENSUS_NO_GBFS"

        for sid in ids:
            mapping[sid] = canonical
        rows.append({
            "id_numeric_key": key,
            "canonical_station_id": canonical,
            "raw_station_ids": ";".join(ids),
            "group_size": len(ids),
            "station_name": group.iloc[0]["station_name"],
            "max_distance_m": round(max_dist, 6),
            "canonical_selection": selection,
            "group_total_activity": int(group["total_activity"].sum()),
        })

    id_mapping = pd.DataFrame([
        {
            "raw_station_id": sid,
            "canonical_station_id": canonical,
            "changed": sid != canonical,
            "mapping_type": "FORMAT_ALIAS" if sid != canonical else "UNCHANGED",
        }
        for sid, canonical in sorted(mapping.items())
    ])
    groups = pd.DataFrame(rows, columns=[
        "id_numeric_key", "canonical_station_id", "raw_station_ids", "group_size",
        "station_name", "max_distance_m", "canonical_selection", "group_total_activity"
    ])
    review = pd.DataFrame(reviews, columns=[
        "id_numeric_key", "raw_station_ids", "issue", "max_distance_m"
    ])
    return mapping, id_mapping, groups, review


def reaggregate_flow(path: Path, mapping: dict) -> tuple[pd.DataFrame, dict]:
    parts = []
    original_sums = {"start": 0, "end": 0}
    raw_rows = 0
    for chunk in pd.read_csv(
        path,
        dtype={"station_id": str, "date": str, "type": str},
        chunksize=250000,
    ):
        raw_rows += len(chunk)
        if chunk["station_id"].isna().any():
            raise ValueError("流量表存在空站点 ID")
        if not chunk["station_id"].isin(mapping).all():
            raise ValueError("流量表含有未出现在登记表中的站点")
        chunk["count"] = pd.to_numeric(chunk["count"], errors="raise").astype("int64")
        if (chunk["count"] < 0).any() or not chunk["type"].isin(["start", "end"]).all():
            raise ValueError("流量数据的 count/type 非法")
        for typ in original_sums:
            original_sums[typ] += int(chunk.loc[chunk["type"] == typ, "count"].sum())
        chunk["station_id"] = chunk["station_id"].map(mapping)
        parts.append(chunk.groupby(
            ["station_id", "date", "hour", "type"], as_index=False, sort=False
        )["count"].sum())
    if not parts:
        raise ValueError("小时流量输入为空")
    merged = pd.concat(parts, ignore_index=True).groupby(
        ["station_id", "date", "hour", "type"], as_index=False, sort=False
    )["count"].sum()
    merged["hour"] = pd.to_numeric(merged["hour"]).astype(int)
    merged = merged.sort_values(["date", "hour", "station_id", "type"]).reset_index(drop=True)
    fixed_sums = merged.groupby("type")["count"].sum().to_dict()
    if any(original_sums[t] != int(fixed_sums.get(t, 0)) for t in original_sums):
        raise AssertionError("小时借还事件总量发生变化")
    return merged, {"original_rows": raw_rows, "fixed_rows": len(merged), **original_sums}


def reaggregate_od(path: Path, mapping: dict) -> tuple[pd.DataFrame, int, int]:
    pieces = []
    total_before = 0
    for chunk in pd.read_csv(
        path,
        dtype={"start_station_id": str, "end_station_id": str},
        chunksize=250000,
    ):
        if not chunk["start_station_id"].isin(mapping).all() or not chunk["end_station_id"].isin(mapping).all():
            raise ValueError("OD 表含有未出现在登记表中的站点")
        chunk["trips"] = pd.to_numeric(chunk["trips"], errors="raise").astype("int64")
        if (chunk["trips"] < 0).any():
            raise ValueError("OD 存在负骑行次数")
        total_before += int(chunk["trips"].sum())
        for col in ["start_station_id", "end_station_id"]:
            chunk[col] = chunk[col].map(mapping)
        pieces.append(chunk.groupby(
            ["start_station_id", "end_station_id"], as_index=False, sort=False
        )["trips"].sum())
    if not pieces:
        raise ValueError("OD 数据为空")
    merged = pd.concat(pieces, ignore_index=True).groupby(
        ["start_station_id", "end_station_id"], as_index=False, sort=False
    )["trips"].sum()
    merged = merged.sort_values("trips", ascending=False).reset_index(drop=True)
    total_after = int(merged["trips"].sum())
    if total_after != total_before:
        raise AssertionError("OD 骑行总量发生变化")
    return merged, total_before, total_after


def reaggregate_registry(raw: pd.DataFrame, flow: pd.DataFrame, mapping: dict, gbfs: dict) -> pd.DataFrame:
    registry = raw.copy()
    registry["station_id"] = registry["station_id"].map(mapping)
    # 经纬度采用历史站点事件总量加权平均。使用原始活动总量，不将站点坐标视为真实历史桩位快照。
    registry["weighted_lat"] = registry["latitude"] * registry["total_activity"]
    registry["weighted_lon"] = registry["longitude"] * registry["total_activity"]
    g = registry.groupby("station_id", as_index=False)
    stats = g.agg(
        weighted_lat=("weighted_lat", "sum"),
        weighted_lon=("weighted_lon", "sum"),
        original_activity=("total_activity", "sum"),
    )
    stats["latitude"] = stats["weighted_lat"] / stats["original_activity"]
    stats["longitude"] = stats["weighted_lon"] / stats["original_activity"]
    names = (
        registry.sort_values("total_activity", ascending=False)
        .drop_duplicates("station_id")[["station_id", "station_name"]]
    )
    stats = stats.merge(names, on="station_id", validate="one_to_one")

    piv = flow.groupby(["station_id", "type"])["count"].sum().unstack(fill_value=0)
    for typ in ["start", "end"]:
        if typ not in piv:
            piv[typ] = 0
    piv = piv.rename(columns={"start": "departures", "end": "arrivals"}).reset_index()
    days = (flow[["station_id", "date"]].drop_duplicates()
            .groupby("station_id").size().rename("active_days").reset_index())
    stats = stats.merge(piv, on="station_id", validate="one_to_one")
    stats = stats.merge(days, on="station_id", validate="one_to_one")
    stats["total_activity"] = stats["departures"] + stats["arrivals"]
    if not (stats["original_activity"] == stats["total_activity"]).all():
        raise AssertionError("原始站点活动量与修复后活动量不一致")

    for col in ["capacity", "metadata_lat", "metadata_lon", "metadata_name", "gbfs_station_id"]:
        stats[col] = None
    for index, row in stats.iterrows():
        sid = row["station_id"]
        if sid not in gbfs:
            continue
        info = gbfs[sid]
        stats.at[index, "capacity"] = info.get("capacity")
        stats.at[index, "metadata_lat"] = info.get("lat")
        stats.at[index, "metadata_lon"] = info.get("lon")
        stats.at[index, "metadata_name"] = info.get("name") if isinstance(info.get("name"), str) else str(info.get("name"))
        stats.at[index, "gbfs_station_id"] = str(info.get("station_id", ""))
    stats["capacity"] = pd.to_numeric(stats["capacity"], errors="coerce")
    stats["capacity_source"] = stats["capacity"].apply(
        lambda c: "CURRENT_GBFS_POSITIVE_NOT_HISTORICAL" if pd.notna(c) and c > 0
        else "CURRENT_GBFS_NONPOSITIVE_NOT_HISTORICAL" if pd.notna(c)
        else "UNAVAILABLE"
    )
    ordered = ["station_id", "station_name", "latitude", "longitude", "active_days",
               "departures", "arrivals", "total_activity", "capacity", "metadata_lat",
               "metadata_lon", "metadata_name", "capacity_source", "gbfs_station_id"]
    return stats[ordered].sort_values("total_activity", ascending=False).reset_index(drop=True)


def convert_reference_clusters(path: Path, mapping: dict) -> pd.DataFrame:
    """只保留原候选网络的映射成员作重叠度参照；不沿用任何旧统计指标。"""
    if not path.is_file():
        return pd.DataFrame(columns=["seed_station_id", "station_ids", "source_note"])
    raw = pd.read_csv(path, dtype={"seed_station_id": str, "station_ids": str})
    out = []
    for row in raw.itertuples(index=False):
        members = sorted(set(mapping.get(s, s) for s in str(row.station_ids).split(";") if s))
        out.append({
            "seed_station_id": mapping.get(row.seed_station_id, row.seed_station_id),
            "station_ids": ";".join(members),
            "source_note": "REFERENCE_ONLY_OLD_CLUSTER_MEMBERS_ID_MAPPED_NO_RERANK",
        })
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description="Day 2 Citi Bike historical station ID repair")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coordinate-tolerance-m", type=float, default=25.0)
    args = parser.parse_args()
    if args.coordinate_tolerance_m <= 0:
        parser.error("坐标容差必须大于 0")
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("输出目录不得等于原始输入目录，以免覆盖原始数据")
    paths = {n: args.input_dir / n for n in [
        "station_registry.csv", "station_flow_summary.csv", "station_od_counts.csv"
    ]}
    for p in [*paths.values(), args.metadata]:
        if not p.is_file():
            raise FileNotFoundError(f"缺少输入: {p}")

    print("[1/5] 读取历史站点和当前 GBFS 元数据...")
    raw = pd.read_csv(paths["station_registry.csv"], dtype={"station_id": str})
    if raw["station_id"].isna().any() or raw["station_id"].duplicated().any():
        raise ValueError("原始站点登记表有空 ID 或重复 ID")
    for col in ["latitude", "longitude", "total_activity", "departures", "arrivals"]:
        raw[col] = pd.to_numeric(raw[col], errors="raise")
    gbfs, gbfs_by_key = load_gbfs(args.metadata)

    mapping, map_df, groups, reviews = make_mapping(
        raw, gbfs_by_key, args.coordinate_tolerance_m
    )
    print(f"发现确认可归并的编号组: {len(groups)}；待人工复核组: {len(reviews)}")
    print("[2/5] 重新聚合小时借还事件...")
    flow, flow_check = reaggregate_flow(paths["station_flow_summary.csv"], mapping)
    print("[3/5] 重新聚合完整 OD 网络...")
    od, trips_before, trips_after = reaggregate_od(paths["station_od_counts.csv"], mapping)
    print("[4/5] 重建站点汇总并执行守恒检查...")
    fixed = reaggregate_registry(raw, flow, mapping, gbfs)

    orig_start = int(raw["departures"].sum())
    orig_end = int(raw["arrivals"].sum())
    start = int(fixed["departures"].sum())
    end = int(fixed["arrivals"].sum())
    if not (orig_start == start == trips_before == trips_after == flow_check["start"]):
        raise AssertionError("全局借出总数不一致")
    if not (orig_end == end == trips_after == flow_check["end"]):
        raise AssertionError("全局归还总数不一致")

    # 每站 OD 出入量必须与聚合站点的借还量一致。
    od_starts = od.groupby("start_station_id")["trips"].sum()
    od_ends = od.groupby("end_station_id")["trips"].sum()
    check = fixed.set_index("station_id")
    if not check["departures"].equals(od_starts.reindex(check.index, fill_value=0).astype("int64")):
        raise AssertionError("站点借出量与 OD 出发量不一致")
    if not check["arrivals"].equals(od_ends.reindex(check.index, fill_value=0).astype("int64")):
        raise AssertionError("站点归还量与 OD 到达量不一致")

    print("[5/5] 保存独立修复副本和审计报告...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(args.output_dir / "station_registry.csv", index=False, encoding="utf-8-sig")
    flow.to_csv(args.output_dir / "station_flow_summary.csv", index=False, encoding="utf-8-sig")
    od.to_csv(args.output_dir / "station_od_counts.csv", index=False, encoding="utf-8-sig")
    map_df.to_csv(args.output_dir / "station_id_mapping.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(args.output_dir / "station_alias_groups.csv", index=False, encoding="utf-8-sig")
    reviews.to_csv(args.output_dir / "station_id_review.csv", index=False, encoding="utf-8-sig")
    original_clusters_path = args.input_dir / "candidate_clusters.csv"
    if original_clusters_path.is_file():
        convert_reference_clusters(original_clusters_path, mapping).to_csv(
            args.output_dir / "candidate_clusters.csv", index=False, encoding="utf-8-sig"
        )

    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(args.input_dir.resolve()),
        "output_directory": str(args.output_dir.resolve()),
        "raw_station_ids": len(raw),
        "canonical_station_ids": len(fixed),
        "confirmed_alias_groups": len(groups),
        "changed_id_count": int(map_df["changed"].sum()),
        "manual_review_groups": len(reviews),
        "trips_before": trips_before,
        "trips_after": trips_after,
        "departures_before": orig_start,
        "departures_after": start,
        "arrivals_before": orig_end,
        "arrivals_after": end,
        "flow_rows_before": flow_check["original_rows"],
        "flow_rows_after": flow_check["fixed_rows"],
        "positive_proxy_capacity_stations": int(fixed["capacity"].gt(0).sum()),
        "note": "Station aliases inferred from numeric formatting + identical name + historical coordinates. Current GBFS is not historical evidence of capacity."
    }
    (args.output_dir / "repair_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = [
        "# Day 2 站点编号修复与守恒验收", "",
        f"- 原始历史站点 ID 数：{len(raw)}",
        f"- 修复后规范站点 ID 数：{len(fixed)}",
        f"- 已归并格式别名组：{len(groups)}",
        f"- 仍需人工复核组：{len(reviews)}",
        f"- 原始 / 修复后完整骑行量：{trips_before:,} / {trips_after:,}",
        f"- 原始 / 修复后借车事件：{orig_start:,} / {start:,}",
        f"- 原始 / 修复后还车事件：{orig_end:,} / {end:,}",
        "- 每站汇总与 OD 出入站流量：全部一致", "",
        "## 原则", "",
        "- 保留原始输入目录，不覆盖源证据。",
        "- 只合并数字格式等价、名称相同、坐标接近的历史站点记录。",
        "- 未通过证据检查的候选映射写入 station_id_review.csv，不自动合并。",
        "- 修复后的 candidate_clusters.csv 仅用于原候选成员重叠度参照，不包含旧的排名指标。",
        "- GBFS 容量来自当前快照，不能视为 2026 年 4 月历史容量。",
        "- 需要重新运行 Day 2-A 才能获得有效的候选区域统计。", ""
    ]
    (args.output_dir / "repair_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"PASS：骑行记录总量={trips_after:,}，借出/归还均守恒")
    print(f"历史站点 ID：{len(raw)} -> {len(fixed)}；归并 {len(groups)} 组")
    print(f"待复核：{len(reviews)} 组；容量代理为正：{int(fixed['capacity'].gt(0).sum())} 站")
    print(f"输出目录：{args.output_dir.resolve()}")
    if len(reviews):
        print("注意：存在未自动修复的冲突组，请查看 station_id_review.csv")


if __name__ == "__main__":
    main()
