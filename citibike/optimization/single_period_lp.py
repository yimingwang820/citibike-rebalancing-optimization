#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 3 / Lesson 6: single-hour Citi Bike rebalancing LP.

Data: corrected Day 2 hourly history and scenario inventories.
Model: all rebalancing happens before borrowing; no arrivals during this hour.
This is a *scenario model*, not a reconstruction of historical operations.

Windows PowerShell (from project directory):
    python -m citibike.optimization.single_period_lp

Requires: gurobipy (already installed in the user's environment).
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import json
import math
from pathlib import Path

from citibike.paths import ROOT


def load_inventory(path: Path, scenario: str) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到库存情景表: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"scenario", "station_id", "station_name", "proxy_capacity", "initial_inventory"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"库存表缺少列: {sorted(missing)}")
        rows = [r for r in reader if r["scenario"] == scenario]
    if len(rows) != 25:
        raise ValueError(f"情景 {scenario}: 应有25站，实际为 {len(rows)}")
    seen = set()
    for r in rows:
        sid = r["station_id"].strip()
        if not sid or sid in seen:
            raise ValueError(f"站点 ID 重复或缺失: {sid}")
        seen.add(sid)
        cap = float(r["proxy_capacity"])
        init = float(r["initial_inventory"])
        if not (math.isfinite(cap) and cap > 0 and cap.is_integer()
                and math.isfinite(init) and init.is_integer() and 0 <= init <= cap):
            raise ValueError(f"库存或容量非法: {sid}")
        r["station_id"] = sid
        r["capacity"] = int(cap)
        r["initial"] = int(init)
    return sorted(rows, key=lambda r: r["station_id"])


def load_station_locations(path: Path, ids: set[str], seed: str) -> dict[str, tuple[float, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到选站表: {path}")
    locations = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["station_id"] not in ids or r.get("candidate_seed", seed) != seed:
                continue
            sid = r["station_id"]
            if sid in locations:
                raise ValueError(f"选站表重复 ID: {sid}")
            lat, lon = float(r["latitude"]), float(r["longitude"])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError(f"非法坐标: {sid}")
            locations[sid] = (lat, lon)
    if set(locations) != ids:
        raise ValueError(f"坐标与库存站点不一致，缺少: {sorted(ids-set(locations))}")
    return locations


def training_weekdays(begin: str, end: str) -> set[str]:
    a, b = date.fromisoformat(begin), date.fromisoformat(end)
    if b < a or b >= date(2026, 4, 21):
        raise ValueError("本课限制训练期在2026-04-01至2026-04-20内，不使用测试期数据")
    result = set()
    current = a
    while current <= b:
        if current.weekday() < 5:
            result.add(current.isoformat())
        current += timedelta(days=1)
    if not result:
        raise ValueError("没有训练期工作日")
    return result


def load_average_departures(path: Path, ids: set[str], days: set[str], hour: int) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到修复后的小时借还表: {path}")
    summed = {sid: 0 for sid in ids}
    # CSV是修复后的所有历史站点数据，保留非选区骑行的选区起点事件。
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        needed = {"station_id", "date", "hour", "type", "count"}
        if needed - set(reader.fieldnames or []):
            raise ValueError("小时借还表字段不全")
        for r in reader:
            sid = r["station_id"]
            if sid in ids and r["date"] in days and r["type"] == "start" and int(r["hour"]) == hour:
                n = int(r["count"])
                if n < 0:
                    raise ValueError(f"借出事件为负数: {sid}")
                summed[sid] += n
    # 用全部训练工作日作分母：没有借出事件的站点/日期应计为 0。
    return {sid: summed[sid] / len(days) for sid in ids}


def distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    dlat, dlon = lat2-lat1, lon2-lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * 6371.0088 * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def solve_lp(rows: list[dict], demand: dict[str, float],
             locations: dict[str, tuple[float, float]],
             budget: float, penalty: float, cost_per_km: float):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError("找不到 gurobipy，请确认使用已安装 Gurobi 的 Python 环境") from exc

    ids = [r["station_id"] for r in rows]
    initial = {r["station_id"]: r["initial"] for r in rows}
    capacity = {r["station_id"]: r["capacity"] for r in rows}
    edges = [(i, j) for i in ids for j in ids if i != j]
    costs = {(i, j): distance_km(locations[i], locations[j])*cost_per_km
             for i, j in edges}

    model = gp.Model("CitiBike_SingleHour_LP")
    x = model.addVars(edges, lb=0, vtype=GRB.CONTINUOUS, name="transfer")
    p = model.addVars(ids, lb=0, vtype=GRB.CONTINUOUS, name="post_transfer")
    s = model.addVars(ids, lb=0, vtype=GRB.CONTINUOUS, name="served")
    u = model.addVars(ids, lb=0, vtype=GRB.CONTINUOUS, name="unserved_proxy")
    e = model.addVars(ids, lb=0, vtype=GRB.CONTINUOUS, name="end_stock")

    for i in ids:
        outgoing = gp.quicksum(x[i, j] for j in ids if i != j)
        incoming = gp.quicksum(x[j, i] for j in ids if j != i)
        model.addConstr(outgoing <= initial[i], name=f"origin_{i}")
        model.addConstr(incoming <= capacity[i]-initial[i], name=f"dock_{i}")
        model.addConstr(p[i] == initial[i] + incoming - outgoing, name=f"balance_{i}")
        model.addConstr(p[i] <= capacity[i], name=f"capacity_{i}")
        model.addConstr(s[i]+u[i] == demand[i], name=f"demand_{i}")
        model.addConstr(s[i] <= p[i], name=f"service_{i}")
        model.addConstr(e[i] == p[i]-s[i], name=f"end_stock_{i}")

    model.addConstr(gp.quicksum(x[i, j] for i, j in edges) <= budget,
                    name="transfer_budget")
    model.setObjective(
        gp.quicksum(costs[i, j]*x[i, j] for i, j in edges)
        + penalty*gp.quicksum(u[i] for i in ids), GRB.MINIMIZE
    )
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"Gurobi 没有可行解。状态代码: {model.Status}")

    eps = 1e-5
    station_results = []
    for i in ids:
        assert abs(s[i].X+u[i].X-demand[i]) < eps
        assert abs(p[i].X - (initial[i] + sum(x[j, i].X for j in ids if j != i)
                                - sum(x[i, j].X for j in ids if j != i))) < eps
        assert -eps <= e[i].X <= capacity[i]+eps
        station_results.append({
            "station_id": i,
            "station_name": next(r["station_name"] for r in rows if r["station_id"] == i),
            "proxy_capacity": capacity[i],
            "initial_inventory": initial[i],
            "weekday_mean_departures": round(demand[i], 5),
            "no_transfer_shortage_proxy": round(max(0, demand[i]-initial[i]), 5),
            "transfer_in": round(sum(x[j, i].X for j in ids if j != i), 5),
            "transfer_out": round(sum(x[i, j].X for j in ids if j != i), 5),
            "post_transfer_inventory": round(p[i].X, 5),
            "served_proxy": round(s[i].X, 5),
            "unserved_proxy": round(u[i].X, 5),
            "end_inventory": round(e[i].X, 5),
        })

    transfer_results = [
        {"origin_id": i, "destination_id": j,
         "quantity_LP_continuous": round(x[i, j].X, 5),
         "distance_km": round(costs[i, j]/cost_per_km, 5),
         "cost": round(costs[i, j]*x[i, j].X, 5)}
        for i, j in edges if x[i, j].X > eps
    ]
    total_demand = sum(demand.values())
    base_shortage = sum(max(0, demand[i]-initial[i]) for i in ids)
    transfers = sum(x[i, j].X for i, j in edges)
    new_shortage = sum(u[i].X for i in ids)
    relocation_cost = sum(costs[i, j]*x[i, j].X for i, j in edges)
    assert abs(sum(p[i].X for i in ids)-sum(initial.values())) < eps
    assert abs(sum(e[i].X for i in ids) + sum(s[i].X for i in ids) - sum(initial.values())) < eps
    assert transfers <= budget + eps
    assert model.ObjVal <= penalty*base_shortage + eps  # 不调拨一定可行

    summary = {
        "solve_status": "OPTIMAL" if model.Status == GRB.OPTIMAL else str(model.Status),
        "is_mip": model.IsMIP,
        "station_count": len(ids),
        "initial_bikes": sum(initial.values()),
        "proxy_capacity_total": sum(capacity.values()),
        "average_departure_requests_proxy": round(total_demand, 5),
        "no_transfer_shortage_proxy": round(base_shortage, 5),
        "lp_shortage_proxy": round(new_shortage, 5),
        "shortage_reduction_proxy": round(base_shortage-new_shortage, 5),
        "transfer_budget_bikes": budget,
        "lp_transfer_quantity": round(transfers, 5),
        "unit_cost_per_bike_km": cost_per_km,
        "shortage_penalty_per_request": penalty,
        "relocation_cost": round(relocation_cost, 5),
        "lp_objective": round(model.ObjVal, 5),
        "no_transfer_objective": round(penalty*base_shortage, 5),
        "note": "Average observed weekday completed departures used as proxy; no arrivals, no true unmet demand, no historical stock, no truck routing. LP rebalancing may be fractional.",
    }
    return summary, station_results, transfer_results


def save_results(out: Path, summary: dict, stations: list[dict], moves: list[dict]):
    out.mkdir(parents=True, exist_ok=True)
    def write_csv(path, data, fields):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    write_csv(out/"lp_station_results.csv", stations, list(stations[0]))
    write_csv(out/"lp_transfer_plan.csv", moves,
              ["origin_id", "destination_id", "quantity_LP_continuous", "distance_km", "cost"])
    (out/"lp_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# Day 3：单周期 LP 原型", "", "## 关键指标", ""]
    for k, v in summary.items():
        report.append(f"- {k}: {v}")
    report.extend(["", "## 解释边界", "", "- 训练窗口内工作日同一小时的已完成借车量均值，仅作为需求代理。",
                   "- 忽略该小时内归还，所有运营调拨先于借车完成。",
                   "- 站点容量为当前GBFS快照，初始库存为合成情景，不是真实历史状态。",
                   "- 允许连续调拨量；下一阶段再加入车辆整数性和固定任务启动成本。",
                   "- 两种情景需使用相同参数比较，才能讨论初始库存布局影响。", ""])
    (out/"lp_report.md").write_text("\n".join(report), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Day 3 - 真实站点单小时 Gurobi LP")
    ap.add_argument("--inventory", type=Path,
                    default=ROOT/"02_RESULTS"/"DAY2_B_INVENTORY"/"initial_inventory_scenarios.csv")
    ap.add_argument("--stations", type=Path,
                    default=ROOT/"02_RESULTS"/"DAY2_A_FIXED"/"day2_selected_candidates.csv")
    ap.add_argument("--hourly", type=Path,
                    default=ROOT/"02_RESULTS"/"DAY2_ID_FIXED"/"station_flow_summary.csv")
    ap.add_argument("--output-dir", type=Path, default=ROOT/"02_RESULTS"/"DAY3_LP")
    ap.add_argument("--seed", default="5980.10")
    ap.add_argument("--scenario", default="balanced_50",
                    choices=["low_30", "balanced_50", "high_70", "uneven_50"])
    ap.add_argument("--hour", type=int, default=17, help="训练期工作日的小时 [0,23]")
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--end", default="2026-04-20")
    ap.add_argument("--budget", type=float, default=80.0, help="本时段调拨车辆上界")
    ap.add_argument("--penalty", type=float, default=10.0, help="每单位需求代理缺口的惩罚")
    ap.add_argument("--cost-per-km", type=float, default=1.0, help="每辆每公里的相对成本")
    ap.add_argument("--check-data-only", action="store_true", help="只核对数据与基线，无需 Gurobi")
    args = ap.parse_args()
    if not 0 <= args.hour <= 23:
        ap.error("--hour 应在0到23之间")
    if args.budget < 0 or args.penalty < 0 or args.cost_per_km <= 0:
        ap.error("--budget/--penalty 非负，--cost-per-km 必须为正")

    print("[1/3] 读取冻结的25站库存情景与坐标...")
    rows = load_inventory(args.inventory, args.scenario)
    ids = {r["station_id"] for r in rows}
    locations = load_station_locations(args.stations, ids, args.seed)
    print("[2/3] 用修复后的小时事件构造训练工作日借车需求代理...")
    days = training_weekdays(args.start, args.end)
    demand = load_average_departures(args.hourly, ids, days, args.hour)
    baseline = sum(max(0., demand[r["station_id"]]-r["initial"]) for r in rows)
    print(f"训练工作日: {len(days)} 天；时段: {args.hour:02d}:00-{args.hour+1:02d}:00")
    print(f"初始库存: {sum(r['initial'] for r in rows)}；平均借车需求代理: {sum(demand.values()):.3f}")
    print(f"无调拨缺口代理: {baseline:.3f}；假设调拨上限: {args.budget:.1f}")
    if args.check_data_only:
        print("PASS: 数据检查完成，未调用 Gurobi")
        return
    print("[3/3] 构建并求解 Gurobi LP...")
    summary, stations, moves = solve_lp(rows, demand, locations,
                                       args.budget, args.penalty, args.cost_per_km)
    save_results(args.output_dir, summary, stations, moves)
    print("\n--- Day 3 LP 结果 ---")
    print("Status:", summary["solve_status"], "| IsMIP:", summary["is_mip"])
    print("无调拨缺口代理:", summary["no_transfer_shortage_proxy"])
    print("优化后缺口代理:", summary["lp_shortage_proxy"])
    print("累计调拨量:", summary["lp_transfer_quantity"])
    print("调拨成本:", summary["relocation_cost"])
    print("LP 最优目标值:", summary["lp_objective"])
    print("输出目录:", args.output_dir.resolve())
    if summary["solve_status"] != "OPTIMAL":
        print("警告：找到可行解但未证明 LP 全局最优，请先排查求解状态。")


if __name__ == "__main__":
    main()
