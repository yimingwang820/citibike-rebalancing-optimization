#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 3: identical-data LP vs integer bike-transfer MILP.

Place alongside single_period_lp.py in PROJECT/01_CODE; run from any directory.
Needs Python 3.12 and gurobipy in your own environment.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from single_period_lp import (
    ROOT,
    distance_km,
    load_average_departures,
    load_inventory,
    load_station_locations,
    training_weekdays,
)


def optimize_case(rows, locations, demand, budget, penalty, unit_cost, integer):
    import gurobipy as gp
    from gurobipy import GRB

    ids = [r["station_id"] for r in rows]
    cap = {r["station_id"]: r["capacity"] for r in rows}
    ini = {r["station_id"]: r["initial"] for r in rows}
    edges = [(i, j) for i in ids for j in ids if i != j]
    kms = {(i, j): distance_km(locations[i], locations[j]) for i, j in edges}
    mode = "MILP" if integer else "LP"

    m = gp.Model("CitiBike_" + mode)
    m.Params.OutputFlag = 0
    m.Params.MIPGap = 0.0
    x = m.addVars(edges, lb=0, vtype=GRB.INTEGER if integer else GRB.CONTINUOUS, name="x")
    p = m.addVars(ids, lb=0, vtype=GRB.CONTINUOUS, name="post_stock")
    s = m.addVars(ids, lb=0, vtype=GRB.CONTINUOUS, name="served_proxy")
    u = m.addVars(ids, lb=0, vtype=GRB.CONTINUOUS, name="unserved_proxy")
    e = m.addVars(ids, lb=0, vtype=GRB.CONTINUOUS, name="end_stock_proxy")

    for i in ids:
        outgoing = gp.quicksum(x[i, j] for j in ids if j != i)
        incoming = gp.quicksum(x[j, i] for j in ids if j != i)
        # 不允许在同一调拨阶段转运刚收到的自行车。
        m.addConstr(outgoing <= ini[i], name=f"origin_{i}")
        m.addConstr(incoming <= cap[i] - ini[i], name=f"dock_{i}")
        m.addConstr(p[i] == ini[i] + incoming - outgoing, name=f"balance_{i}")
        m.addConstr(p[i] <= cap[i], name=f"capacity_{i}")
        m.addConstr(s[i] + u[i] == demand[i], name=f"demand_{i}")
        m.addConstr(s[i] <= p[i], name=f"served_{i}")
        m.addConstr(e[i] == p[i] - s[i], name=f"ending_{i}")

    m.addConstr(gp.quicksum(x[i, j] for i, j in edges) <= budget, name="budget")
    m.setObjective(
        gp.quicksum(unit_cost * kms[i, j] * x[i, j] for i, j in edges)
        + penalty * gp.quicksum(u[i] for i in ids), GRB.MINIMIZE
    )
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"{mode} 未证明最优：Status={m.Status}；不要将当前结果当作最优解")

    moves = []
    for i, j in edges:
        q = x[i, j].X
        if q > 1e-6:
            moves.append({
                "model": mode, "origin_id": i, "destination_id": j,
                "quantity": round(q, 6), "distance_km": round(kms[i, j], 6),
                "transfer_cost": round(unit_cost * kms[i, j] * q, 6),
            })
    moves.sort(key=lambda r: (-r["quantity"], r["origin_id"], r["destination_id"]))

    total_moves = sum(x[i, j].X for i, j in edges)
    relocation_cost = sum(unit_cost * kms[i, j] * x[i, j].X for i, j in edges)
    shortage = sum(u[i].X for i in ids)
    base_shortage = sum(max(0.0, demand[i] - ini[i]) for i in ids)
    tol = 1e-5
    if abs(sum(p[i].X for i in ids) - sum(ini.values())) > tol:
        raise AssertionError("调拨前后车辆守恒失败")
    if abs(m.ObjVal - (relocation_cost + penalty * shortage)) > tol:
        raise AssertionError("目标函数复核失败")
    if total_moves > budget + tol or shortage > base_shortage + tol:
        raise AssertionError("调拨预算或无调拨基线验收失败")
    if integer and any(abs(r["quantity"] - round(r["quantity"])) > tol for r in moves):
        raise AssertionError("整数调拨变量出现小数辆自行车")

    result = {
        "model": mode,
        "status": "OPTIMAL",
        "objective": round(m.ObjVal, 6),
        "shortage_proxy": round(shortage, 6),
        "shortage_reduction_proxy": round(base_shortage - shortage, 6),
        "transfer_bikes": round(total_moves, 6),
        "transfer_cost": round(relocation_cost, 6),
        "nonzero_routes": len(moves),
        "fractional_routes": sum(abs(x[i, j].X - round(x[i, j].X)) > tol
                                 for i, j in edges if x[i, j].X > tol),
        "run_seconds": round(m.Runtime, 4),
    }
    return result, moves


def write_csv(path, rows, names):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Day 3: LP 和整数调拨 MILP 同数据对照")
    parser.add_argument("--inventory", type=Path, default=ROOT / "02_RESULTS/DAY2_B_INVENTORY/initial_inventory_scenarios.csv")
    parser.add_argument("--stations", type=Path, default=ROOT / "02_RESULTS/DAY2_A_FIXED/day2_selected_candidates.csv")
    parser.add_argument("--hourly", type=Path, default=ROOT / "02_RESULTS/DAY2_ID_FIXED/station_flow_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "02_RESULTS/DAY3_LP_MILP")
    parser.add_argument("--seed", default="5980.10")
    parser.add_argument("--scenario", choices=["low_30", "balanced_50", "high_70", "uneven_50"], default="balanced_50")
    parser.add_argument("--hour", type=int, default=17)
    parser.add_argument("--budget", type=float, default=80.0)
    parser.add_argument("--penalty", type=float, default=10.0)
    parser.add_argument("--cost-per-km", type=float, default=1.0)
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-04-20")
    parser.add_argument("--check-data-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.hour <= 23 or args.budget < 0 or args.penalty < 0 or args.cost_per_km <= 0:
        parser.error("参数不合理：hour须为0-23，budget/penalty非负，cost-per-km为正")

    print("[1/3] 读取相同的站点、库存与训练期需求...")
    rows = load_inventory(args.inventory, args.scenario)
    ids = {r["station_id"] for r in rows}
    locations = load_station_locations(args.stations, ids, args.seed)
    weekdays = training_weekdays(args.start, args.end)
    demand = load_average_departures(args.hourly, ids, weekdays, args.hour)
    base_shortage = sum(max(0.0, demand[r["station_id"]] - r["initial"]) for r in rows)
    print(f"场景={args.scenario}，训练工作日={len(weekdays)}，小时={args.hour}，总库存={sum(r['initial'] for r in rows)}")
    print(f"需求代理={sum(demand.values()):.5f}，不调拨缺口代理={base_shortage:.5f}")
    if args.check_data_only:
        print("PASS：数据检查通过，未求解 Gurobi")
        return

    print("[2/3] 使用连续调拨量求解 LP...")
    lp, lp_moves = optimize_case(rows, locations, demand, args.budget, args.penalty, args.cost_per_km, integer=False)
    print("[3/3] 仅将调拨量改为非负整数，求解 MILP...")
    milp, milp_moves = optimize_case(rows, locations, demand, args.budget, args.penalty, args.cost_per_km, integer=True)

    if lp["objective"] > milp["objective"] + 1e-4:
        raise AssertionError("违反 LP 松弛下界关系：LP 最优值不能超过 MILP 最优值")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "lp_vs_milp_summary.csv", [lp, milp], list(lp))
    write_csv(args.output_dir / "lp_vs_milp_transfer_plan.csv", lp_moves + milp_moves,
              ["model", "origin_id", "destination_id", "quantity", "distance_km", "transfer_cost"])
    report = {
        "scenario": args.scenario, "hour": args.hour, "training_weekdays": len(weekdays),
        "initial_bikes": sum(r["initial"] for r in rows), "departure_proxy": sum(demand.values()),
        "baseline_shortage_proxy": base_shortage, "transfer_budget": args.budget,
        "penalty": args.penalty, "cost_per_km": args.cost_per_km,
        "lp": lp, "milp": milp,
        "absolute_integrality_gap": milp["objective"] - lp["objective"],
        "relative_integrality_gap_vs_milp": (milp["objective"] - lp["objective"]) / max(1e-9, abs(milp["objective"])),
        "caveat": "平均已完成订单是需求代理；s/u保留连续性；调拨先于借车；不计小时内归还与货车路由"
    }
    (args.output_dir / "lp_vs_milp_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n---- LP vs MILP ----")
    for r in [lp, milp]:
        print(f"{r['model']:4s}  目标={r['objective']:.5f}  缺口={r['shortage_proxy']:.5f}  "
              f"调拨={r['transfer_bikes']:.2f}  调拨成本={r['transfer_cost']:.5f}  "
              f"采用路线={r['nonzero_routes']}  非整数路线={r['fractional_routes']}")
    print(f"整数性带来的目标值差距（MILP-LP）: {report['absolute_integrality_gap']:.6f}")
    print("输出目录:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
