
# -*- coding: utf-8 -*-
"""
Day 3：Citi Bike 固定调拨任务成本 MILP

实验 A：整数调拨，无固定任务成本
实验 B：整数调拨 + 固定任务成本
实验 C：整数调拨 + 固定任务成本 + 路线数量限制

依赖：
    gurobipy
    single_period_lp.py（上一节已经运行成功）

运行：
    直接在 VS Code 中运行本文件即可。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from citibike.optimization.single_period_lp import (
    ROOT,
    distance_km,
    load_average_departures,
    load_inventory,
    load_station_locations,
    training_weekdays,
)


# ============================================================
# 1. 固定实验参数
# ============================================================

SEED = "5980.10"
SCENARIO = "balanced_50"

START_DATE = "2026-04-01"
END_DATE = "2026-04-20"
TARGET_HOUR = 17

TRANSFER_BUDGET = 80
SHORTAGE_PENALTY = 10.0
COST_PER_KM = 1.0

FIXED_COST = 3.0
MAX_ROUTES = 4

TIME_LIMIT = 90
MIP_GAP = 0.001

DATA_DIR = ROOT / "02_RESULTS"

INVENTORY_FILE = (
    DATA_DIR
    / "DAY2_B_INVENTORY"
    / "initial_inventory_scenarios.csv"
)

STATIONS_FILE = (
    DATA_DIR
    / "DAY2_A_FIXED"
    / "day2_selected_candidates.csv"
)

HOURLY_FILE = (
    DATA_DIR
    / "DAY2_ID_FIXED"
    / "station_flow_summary.csv"
)

OUTPUT_DIR = DATA_DIR / "DAY3_FIXED_CHARGE"


# ============================================================
# 2. 读取数据
# ============================================================

def load_problem_data():

    print("[1/4] 读取站点、库存和借车需求...")

    rows = load_inventory(
        INVENTORY_FILE,
        SCENARIO
    )

    ids = [r["station_id"] for r in rows]

    locations = load_station_locations(
        STATIONS_FILE,
        set(ids),
        SEED
    )

    days = training_weekdays(
        START_DATE,
        END_DATE
    )

    demand = load_average_departures(
        HOURLY_FILE,
        set(ids),
        days,
        TARGET_HOUR
    )

    initial = {
        r["station_id"]: r["initial"]
        for r in rows
    }

    capacity = {
        r["station_id"]: r["capacity"]
        for r in rows
    }

    baseline_shortage = sum(
        max(0, demand[i] - initial[i])
        for i in ids
    )

    total_inventory = sum(initial.values())
    total_demand = sum(demand.values())

    print(f"研究站点数：{len(ids)}")
    print(f"训练工作日：{len(days)}")
    print(f"研究时段：{TARGET_HOUR}:00")
    print(f"初始库存：{total_inventory}")
    print(f"需求代理量：{total_demand:.5f}")
    print(f"无调拨缺口：{baseline_shortage:.5f}")

    assert len(ids) == 25
    assert len(set(ids)) == 25

    return {
        "ids": ids,
        "initial": initial,
        "capacity": capacity,
        "locations": locations,
        "demand": demand,
        "baseline_shortage": baseline_shortage,
    }


# ============================================================
# 3. 构建并求解 MILP
# ============================================================

def solve_model(
    data,
    case_name,
    fixed_cost,
    max_routes=None
):

    ids = data["ids"]
    initial = data["initial"]
    capacity = data["capacity"]
    locations = data["locations"]
    demand = data["demand"]

    # 每条有向调拨边的最大可调拨整数数量
    max_budget = math.floor(
        TRANSFER_BUDGET
    )

    edges = []

    big_m = {}

    for i in ids:
        for j in ids:

            if i == j:
                continue

            upper = min(
                initial[i],
                capacity[j] - initial[j],
                max_budget
            )

            if upper >= 1:

                edges.append((i, j))
                big_m[i, j] = int(upper)

    # 站点之间的距离和单位成本
    distance = {
        (i, j): distance_km(
            locations[i],
            locations[j]
        )
        for i, j in edges
    }

    cost = {
        (i, j): COST_PER_KM * distance[i, j]
        for i, j in edges
    }

    outgoing = {
        i: [j for a, j in edges if a == i]
        for i in ids
    }

    incoming = {
        i: [j for j, a in edges if a == i]
        for i in ids
    }

    model = gp.Model(case_name)

    model.Params.OutputFlag = 0
    model.Params.TimeLimit = TIME_LIMIT
    model.Params.MIPGap = MIP_GAP

    # --------------------------------------------------------
    # 3.1 决策变量
    # --------------------------------------------------------

    # x_ij：从 i 到 j 调拨的整数车辆数
    x = model.addVars(
        edges,
        lb=0,
        vtype=GRB.INTEGER,
        name="transfer"
    )

    # p_i：调拨后的库存
    p = model.addVars(
        ids,
        lb=0,
        vtype=GRB.CONTINUOUS,
        name="post_inventory"
    )

    # s_i：被满足的需求代理量
    s = model.addVars(
        ids,
        lb=0,
        vtype=GRB.CONTINUOUS,
        name="served"
    )

    # u_i：未被满足的需求代理量
    u = model.addVars(
        ids,
        lb=0,
        vtype=GRB.CONTINUOUS,
        name="shortage"
    )

    # e_i：借车后的剩余库存
    e = model.addVars(
        ids,
        lb=0,
        vtype=GRB.CONTINUOUS,
        name="ending_inventory"
    )

    # 模型 B、C 才需要任务启用变量
    use_binary = (
        fixed_cost > 0
        or max_routes is not None
    )

    if use_binary:

        y = model.addVars(
            edges,
            vtype=GRB.BINARY,
            name="route_open"
        )

    else:

        y = None

    # --------------------------------------------------------
    # 3.2 库存、容量、需求约束
    # --------------------------------------------------------

    for i in ids:

        total_out = gp.quicksum(
            x[i, j]
            for j in outgoing[i]
        )

        total_in = gp.quicksum(
            x[j, i]
            for j in incoming[i]
        )

        # 不能调出超过期初拥有的车辆
        model.addConstr(
            total_out <= initial[i],
            name=f"outbound_{i}"
        )

        # 调入不能超过期初空余车位
        model.addConstr(
            total_in <= capacity[i] - initial[i],
            name=f"inbound_{i}"
        )

        # 调拨库存守恒
        model.addConstr(
            p[i] == (
                initial[i]
                + total_in
                - total_out
            ),
            name=f"balance_{i}"
        )

        # 站点容量约束
        model.addConstr(
            p[i] <= capacity[i],
            name=f"capacity_{i}"
        )

        # 需求分解
        model.addConstr(
            s[i] + u[i] == demand[i],
            name=f"demand_{i}"
        )

        # 满足需求不能超过现有车辆
        model.addConstr(
            s[i] <= p[i],
            name=f"service_{i}"
        )

        # 借车后的库存
        model.addConstr(
            e[i] == p[i] - s[i],
            name=f"ending_{i}"
        )

    # --------------------------------------------------------
    # 3.3 总调拨车辆数限制
    # --------------------------------------------------------

    model.addConstr(
        gp.quicksum(
            x[i, j]
            for i, j in edges
        ) <= TRANSFER_BUDGET,
        name="transfer_budget"
    )

    # --------------------------------------------------------
    # 3.4 固定成本与二元变量
    # --------------------------------------------------------

    if use_binary:

        for i, j in edges:

            # 未启用路线则不能调拨
            model.addConstr(
                x[i, j]
                <= big_m[i, j] * y[i, j],
                name=f"route_upper_{i}_{j}"
            )

            # 启用路线则至少调拨一辆车
            model.addConstr(
                x[i, j] >= y[i, j],
                name=f"route_lower_{i}_{j}"
            )

        # 模型 C 的路线数量约束
        if max_routes is not None:

            model.addConstr(
                gp.quicksum(
                    y[i, j]
                    for i, j in edges
                ) <= max_routes,
                name="max_routes"
            )

    # --------------------------------------------------------
    # 3.5 目标函数
    # --------------------------------------------------------

    transport_expr = gp.quicksum(
        cost[i, j] * x[i, j]
        for i, j in edges
    )

    if use_binary:

        fixed_expr = fixed_cost * gp.quicksum(
            y[i, j]
            for i, j in edges
        )

    else:

        fixed_expr = 0

    shortage_expr = SHORTAGE_PENALTY * gp.quicksum(
        u[i]
        for i in ids
    )

    model.setObjective(
        transport_expr
        + fixed_expr
        + shortage_expr,
        GRB.MINIMIZE
    )

    # --------------------------------------------------------
    # 3.6 求解
    # --------------------------------------------------------

    model.optimize()

    if model.SolCount == 0:

        raise RuntimeError(
            f"{case_name} 未找到可行解，"
            f"Gurobi Status={model.Status}"
        )

    if model.Status == GRB.OPTIMAL:
        status = "OPTIMAL"

    elif model.Status == GRB.TIME_LIMIT:
        status = "TIME_LIMIT_WITH_INCUMBENT"

    else:
        status = f"STATUS_{model.Status}_WITH_INCUMBENT"

    # --------------------------------------------------------
    # 3.7 提取结果
    # --------------------------------------------------------

    moves = []

    for i, j in edges:

        quantity = x[i, j].X

        if quantity < 1e-5:
            continue

        if abs(quantity - round(quantity)) > 1e-5:
            raise AssertionError("调拨出现小数辆车")

        if use_binary:
            if abs(y[i, j].X - 1) > 1e-5:
                raise AssertionError(
                    "使用了未启用的路线"
                )

        quantity = int(round(quantity))

        variable_cost = (
            cost[i, j] * quantity
        )

        fixed_route_cost = (
            fixed_cost if use_binary else 0
        )

        moves.append({
            "case": case_name,
            "origin_id": i,
            "destination_id": j,
            "bikes": quantity,
            "distance_km": round(
                distance[i, j], 6
            ),
            "transport_cost": round(
                variable_cost, 6
            ),
            "fixed_cost": round(
                fixed_route_cost, 6
            ),
            "total_cost": round(
                variable_cost + fixed_route_cost,
                6
            )
        })

    moves.sort(
        key=lambda r: (
            -r["bikes"],
            r["origin_id"],
            r["destination_id"]
        )
    )

    total_transfer = sum(
        x[i, j].X
        for i, j in edges
    )

    transport_cost = sum(
        cost[i, j] * x[i, j].X
        for i, j in edges
    )

    if use_binary:

        total_fixed_cost = fixed_cost * sum(
            y[i, j].X
            for i, j in edges
        )

    else:

        total_fixed_cost = 0

    shortage = sum(
        u[i].X
        for i in ids
    )

    objective_check = (
        transport_cost
        + total_fixed_cost
        + SHORTAGE_PENALTY * shortage
    )

    # --------------------------------------------------------
    # 3.8 模型验收
    # --------------------------------------------------------

    if abs(model.ObjVal - objective_check) > 1e-4:
        raise AssertionError(
            "目标函数分解不一致"
        )

    total_after_transfer = sum(
        p[i].X
        for i in ids
    )

    if abs(
        total_after_transfer - sum(initial.values())
    ) > 1e-4:
        raise AssertionError(
            "调拨后库存总量不守恒"
        )

    if total_transfer > TRANSFER_BUDGET + 1e-5:
        raise AssertionError(
            "调拨超出预算"
        )

    if (
        max_routes is not None
        and len(moves) > max_routes
    ):
        raise AssertionError(
            "调拨路线数超出限制"
        )

    if use_binary:

        opened_count = sum(
            y[i, j].X
            for i, j in edges
        )

        if abs(opened_count - len(moves)) > 1e-5:
            raise AssertionError(
                "启用路线数与调拨路线数不一致"
            )

    # --------------------------------------------------------
    # 3.9 汇总
    # --------------------------------------------------------

    summary = {
        "case": case_name,
        "status": status,
        "objective": round(model.ObjVal, 6),
        "shortage_proxy": round(shortage, 6),
        "shortage_reduction": round(
            data["baseline_shortage"] - shortage,
            6
        ),
        "transfer_bikes": int(
            round(total_transfer)
        ),
        "transport_cost": round(
            transport_cost, 6
        ),
        "route_fixed_cost": round(
            total_fixed_cost, 6
        ),
        "total_relocation_cost": round(
            transport_cost + total_fixed_cost,
            6
        ),
        "active_routes": len(moves),
        "max_routes": (
            max_routes
            if max_routes is not None
            else ""
        ),
        "fixed_cost_per_route": fixed_cost,
        "best_bound": round(
            model.ObjBound, 6
        ),
        "mip_gap": round(
            model.MIPGap, 8
        ),
        "runtime_seconds": round(
            model.Runtime, 4
        ),
        "nodes": round(
            model.NodeCount, 2
        )
    }

    return summary, moves


# ============================================================
# 4. 保存 CSV
# ============================================================

def save_csv(path, records):

    if not records:
        return

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)


# ============================================================
# 5. 主程序
# ============================================================

def main():

    data = load_problem_data()

    experiments = [
        {
            "name": "A_integer_only",
            "fixed_cost": 0.0,
            "max_routes": None,
        },
        {
            "name": "B_fixed_charge",
            "fixed_cost": FIXED_COST,
            "max_routes": None,
        },
        {
            "name": "C_fixed_plus_route_limit",
            "fixed_cost": FIXED_COST,
            "max_routes": MAX_ROUTES,
        }
    ]

    all_summaries = []
    all_moves = []

    for index, case in enumerate(
        experiments, start=2
    ):

        print()
        print(
            f"[{index}/4] 求解 {case['name']}..."
        )

        summary, moves = solve_model(
            data=data,
            case_name=case["name"],
            fixed_cost=case["fixed_cost"],
            max_routes=case["max_routes"]
        )

        all_summaries.append(summary)
        all_moves.extend(moves)

        print(
            f"状态：{summary['status']}"
        )

        print(
            f"目标值：{summary['objective']:.5f}"
        )

        print(
            f"缺口代理：{summary['shortage_proxy']:.5f}"
        )

        print(
            f"调拨车辆：{summary['transfer_bikes']}"
        )

        print(
            f"采用路线：{summary['active_routes']}"
        )

        print(
            f"运输成本：{summary['transport_cost']:.5f}"
        )

        print(
            f"固定成本：{summary['route_fixed_cost']:.5f}"
        )

        print(
            f"MIP Gap：{summary['mip_gap']:.4%}"
        )

    # --------------------------------------------------------
    # 6. 输出结果
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_csv(
        OUTPUT_DIR / "fixed_charge_summary.csv",
        all_summaries
    )

    save_csv(
        OUTPUT_DIR / "fixed_charge_transfer_plans.csv",
        all_moves
    )

    audit = {
        "scenario": SCENARIO,
        "seed": SEED,
        "hour": TARGET_HOUR,
        "training_weekdays": len(
            training_weekdays(
                START_DATE, END_DATE
            )
        ),
        "initial_inventory": sum(
            data["initial"].values()
        ),
        "total_capacity": sum(
            data["capacity"].values()
        ),
        "total_demand_proxy": sum(
            data["demand"].values()
        ),
        "baseline_shortage_proxy":
            data["baseline_shortage"],
        "transfer_budget": TRANSFER_BUDGET,
        "shortage_penalty": SHORTAGE_PENALTY,
        "cost_per_km": COST_PER_KM,
        "fixed_cost": FIXED_COST,
        "max_routes": MAX_ROUTES,
        "results": all_summaries,
        "limitations": [
            "借车需求为训练期已完成订单的小时平均值。",
            "初始库存为合成情景。",
            "容量采用当前 GBFS 代理容量。",
            "所有调拨发生在借车之前。",
            "未模拟小时内归还、运输时间与在途车辆。",
            "有向调拨关系不等于真实卡车巡回路线。",
            "固定成本和调拨预算均为假设参数。"
        ]
    }

    with open(
        OUTPUT_DIR / "fixed_charge_audit.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            audit,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # 7. 输出简单报告
    # --------------------------------------------------------

    report_lines = [
        "# Day 3：固定任务成本 MILP 实验",
        "",
        "| 模型 | 目标值 | 缺口代理 | 调拨辆数 | 路线数 | 运输成本 | 固定成本 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for r in all_summaries:

        report_lines.append(
            f"| {r['case']} "
            f"| {r['objective']:.4f} "
            f"| {r['shortage_proxy']:.4f} "
            f"| {r['transfer_bikes']} "
            f"| {r['active_routes']} "
            f"| {r['transport_cost']:.4f} "
            f"| {r['route_fixed_cost']:.4f} "
            f"| {r['status']} |"
        )

    report_lines += [
        "",
        "## 实验说明",
        "",
        "本实验比较整数调拨、固定成本与路线数量限制。",
        "",
        "需求代理不是实际未满足的客户请求量。",
        "初始库存与固定成本均为假设参数。",
        "调拨关系尚未构成真实的卡车路径规划。",
    ]

    (
        OUTPUT_DIR / "fixed_charge_report.md"
    ).write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print()
    print("=" * 55)
    print("Day 3 固定任务成本实验完成")
    print("=" * 55)

    for r in all_summaries:

        print(
            f"{r['case']}: "
            f"目标={r['objective']:.5f}, "
            f"缺口={r['shortage_proxy']:.5f}, "
            f"调拨={r['transfer_bikes']}, "
            f"路线={r['active_routes']}, "
            f"状态={r['status']}"
        )

    print()
    print("结果保存在：", OUTPUT_DIR)

    print(
        "注意：若模型达到时间上限，"
        "不能直接宣称已证明全局最优。"
    )


if __name__ == "__main__":
    main()
