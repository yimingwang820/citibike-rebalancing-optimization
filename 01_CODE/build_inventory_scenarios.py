#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 2-B: build reproducible initial-inventory scenarios for Citi Bike.

Inputs: DAY2_A_FIXED/day2_selected_candidates.csv (the *fixed-ID* selection output).
No optimization, demand prediction, or historical inventory reconstruction is performed.
Requires only the Python standard library.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "02_RESULTS" / "DAY2_A_FIXED" / "day2_selected_candidates.csv"
DEFAULT_OUTPUT = ROOT / "02_RESULTS" / "DAY2_B_INVENTORY"


def rounded_fraction(total: int, fraction: str) -> int:
    """Half-up rounding, avoiding inconsistent per-station rounding."""
    value = Decimal(total) * Decimal(fraction)
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def read_stations(path: Path, seed: str) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到选站文件: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"station_id", "station_name", "proxy_capacity", "distance_to_seed_km"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"缺少字段: {sorted(required - set(reader.fieldnames or []))}")
        all_rows = list(reader)

    if "candidate_seed" in (reader.fieldnames or []):
        rows = [r for r in all_rows if (r.get("candidate_seed") or "").strip() == seed]
    else:
        rows = all_rows

    if len(rows) != 25:
        raise ValueError(f"主实验区域必须恰好有 25 站，当前找到 {len(rows)} 站；seed={seed}")
    seen = set()
    for row in rows:
        sid = row["station_id"].strip()
        if not sid or sid in seen:
            raise ValueError(f"站点 ID 缺失或重复: {sid!r}")
        seen.add(sid)
        cap = float(row["proxy_capacity"])
        if not math.isfinite(cap) or cap <= 0 or not cap.is_integer():
            raise ValueError(f"站点 {sid} 代理容量必须为正整数，实际为 {cap}")
        d = float(row["distance_to_seed_km"])
        if not math.isfinite(d) or d < 0:
            raise ValueError(f"站点 {sid} 距离无效: {d}")
        row["station_id"] = sid
        row["capacity_int"] = int(cap)
        row["distance_num"] = d

    rows.sort(key=lambda r: (r["distance_num"], r["station_id"]))
    return rows


def allocate_proportionally(rows: list[dict], target: int) -> dict[str, int]:
    """Integer largest-remainder allocation: sum stock=target and stock<=capacity."""
    capacity_total = sum(r["capacity_int"] for r in rows)
    if not 0 <= target <= capacity_total:
        raise ValueError(f"车辆总数 {target} 超出可行范围 [0,{capacity_total}]")

    stock, remainders = {}, []
    for r in rows:
        sid, cap = r["station_id"], r["capacity_int"]
        base, remainder = divmod(target * cap, capacity_total)
        stock[sid] = base
        remainders.append((-remainder, sid))

    missing = target - sum(stock.values())
    for _, sid in sorted(remainders)[:missing]:
        stock[sid] += 1

    assert sum(stock.values()) == target
    assert all(0 <= stock[r["station_id"]] <= r["capacity_int"] for r in rows)
    return stock


def make_scenarios(rows: list[dict]) -> tuple[dict, dict]:
    total_capacity = sum(r["capacity_int"] for r in rows)
    count = {
        "low_30": rounded_fraction(total_capacity, "0.30"),
        "balanced_50": rounded_fraction(total_capacity, "0.50"),
        "high_70": rounded_fraction(total_capacity, "0.70"),
    }
    stocks = {
        name: allocate_proportionally(rows, n)
        for name, n in count.items()
    }

    # Radial mismatch: central 12 stations are intentionally relatively empty;
    # outer 13 are relatively full.  Total bikes identical to balanced_50.
    middle = len(rows) // 2
    inner, outer = rows[:middle], rows[middle:]
    total50 = count["balanced_50"]
    inner_target = rounded_fraction(sum(r["capacity_int"] for r in inner), "0.30")
    outer_target = total50 - inner_target
    if not 0 <= outer_target <= sum(r["capacity_int"] for r in outer):
        raise ValueError("外圈容量无法接纳剩余车辆，请调整不均衡情景的设定")

    uneven = allocate_proportionally(inner, inner_target)
    uneven.update(allocate_proportionally(outer, outer_target))
    stocks["uneven_50"] = uneven
    count["uneven_50"] = total50

    inner_ids = {r["station_id"] for r in inner}
    half = stocks["balanced_50"]
    relocated = sum(
        max(0, half[sid] - uneven[sid]) for sid in inner_ids
    )
    # For two-group pure redistribution, units lost inside equal units gained outside.
    assert relocated == sum(max(0, uneven[r["station_id"]] - half[r["station_id"]]) for r in outer)

    info = {
        "main_seed": "5980.10",
        "station_count": len(rows),
        "capacity_total_proxy": total_capacity,
        "scenario_counts": count,
        "spatial_group_rule": "12 stations nearest seed are inner; other 13 are outer",
        "inner_station_count": len(inner),
        "outer_station_count": len(outer),
        "inner_proxy_capacity": sum(r["capacity_int"] for r in inner),
        "outer_proxy_capacity": sum(r["capacity_int"] for r in outer),
        "inner_inventory_uneven_50": inner_target,
        "outer_inventory_uneven_50": outer_target,
        "relocated_bikes_vs_balanced_50": relocated,
        "capacity_is_historical": False,
        "initial_inventory_is_observed": False,
        "warning": "Scenario design only; not historical bike stocks, actual latent demand, or observed unmet demand.",
    }
    groups = {r["station_id"]: ("inner" if r["station_id"] in inner_ids else "outer") for r in rows}
    return stocks, {"audit": info, "groups": groups}


def save_outputs(rows: list[dict], stocks: dict, info: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    output_csv = out / "initial_inventory_scenarios.csv"
    columns = [
        "scenario", "station_id", "station_name", "proxy_capacity",
        "initial_inventory", "free_docks", "initial_fill_ratio",
        "spatial_group", "distance_to_seed_km", "capacity_source",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for scenario in ("low_30", "balanced_50", "high_70", "uneven_50"):
            for r in rows:
                sid, cap = r["station_id"], r["capacity_int"]
                n = stocks[scenario][sid]
                writer.writerow({
                    "scenario": scenario,
                    "station_id": sid,
                    "station_name": r["station_name"],
                    "proxy_capacity": cap,
                    "initial_inventory": n,
                    "free_docks": cap - n,
                    "initial_fill_ratio": f"{n / cap:.6f}",
                    "spatial_group": info["groups"][sid],
                    "distance_to_seed_km": f"{r['distance_num']:.6f}",
                    "capacity_source": r.get("capacity_source", "CURRENT_GBFS_NOT_HISTORICAL"),
                })

    audit_file = out / "inventory_scenario_audit.json"
    audit_file.write_text(
        json.dumps(info["audit"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    a = info["audit"]
    report = [
        "# Day 2-B 初始库存情景", "",
        "## 实验性质", "",
        "- 所有库存都是合成初始条件，并非历史真实库存。",
        "- 站点容量采用当前 GBFS proxy capacity，不是 2026-04 历史容量。",
        "- 观测已完成行程不代表全部潜在需求。",
        "- 50% 与不均衡 50% 的系统车辆总数严格相等。", "",
        "## 容量与初始车辆总数", "",
        f"- 站点数：{a['station_count']}",
        f"- 代理容量总和：{a['capacity_total_proxy']}",
    ]
    for scenario, n in a["scenario_counts"].items():
        report.append(f"- {scenario}：{n} 辆")
    report += [
        "", "## 异质初始库存", "",
        f"- 内圈 {a['inner_station_count']} 站，容量 {a['inner_proxy_capacity']}，初始库存 {a['inner_inventory_uneven_50']}。",
        f"- 外圈 {a['outer_station_count']} 站，容量 {a['outer_proxy_capacity']}，初始库存 {a['outer_inventory_uneven_50']}。",
        f"- 相对均匀 50% 方案，向外圈净转移 {a['relocated_bikes_vs_balanced_50']} 辆。",
        "", "## 必须遵守的约束", "",
        "- 每站：0 <= 初始库存 <= 代理容量。",
        "- 场景：总初始库存等于场景指定的总车辆数。",
        "- 本文件不能推断历史补车、真实缺车或调拨效率。",
    ]
    (out / "inventory_scenario_report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(f"PASS：{a['station_count']} 个站点，代理容量总和 {a['capacity_total_proxy']}")
    for s,n in a["scenario_counts"].items():
        print(f"  {s:12s}: {n:4d} 辆")
    print("不均衡 50% 相对均匀 50% 的净空间转移:", a["relocated_bikes_vs_balanced_50"], "辆")
    print("输出目录:", out.resolve())


def main():
    parser = argparse.ArgumentParser(description="Generate Citi Bike synthetic initial inventory scenarios")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", default="5980.10")
    args = parser.parse_args()
    rows = read_stations(args.input_file, args.seed)
    stocks, info = make_scenarios(rows)
    info["audit"]["main_seed"] = args.seed
    save_outputs(rows, stocks, info, args.output_dir)


if __name__ == "__main__":
    main()
