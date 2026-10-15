
# -*- coding: utf-8 -*-
"""
Day 4-D：先借后还的保守分阶段多周期 MILP。

比较：
1. 无站间调拨
2. 允许站间调拨

时间顺序：
调拨 -> 借车（允许按需外部补车）
     -> 归还（允许同步外部疏导）

直接运行，不需要命令行参数。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

import hourly_balance_milp as previous


# ==================================================
# 1. 继承已验收的数据与实验参数
# ==================================================

OUTPUT = previous.BASE / "DAY4_STRICT_ORDER"

HOURS = previous.HOURS
MAX_MOVES = previous.MOVE_BUDGET

FIXED_COST = previous.FIXED_COST_PER_ARC_HOUR
DISTANCE_COST = previous.COST_PER_BIKE_KM

EXTRA_COST = previous.EXTRA_COST
REMOVE_COST = previous.REMOVE_COST

TIME_LIMIT = 90
MIP_GAP = 0.001
TOL = 1e-5


# ==================================================
# 2. 构建并求解模型
# ==================================================

def solve(data, with_moves):

    (
        ids,
        capacity,
        initial,
        dep,
        arr,
        edges,
        distance
    ) = data

    mode = (
        "with_rebalancing"
        if with_moves
        else "without_rebalancing"
    )

    model = gp.Model(mode + "_strict_order")

    model.Params.OutputFlag = 0
    model.Params.TimeLimit = TIME_LIMIT

    if with_moves:
        model.Params.MIPGap = MIP_GAP

    steps = range(len(HOURS))
    states = range(len(HOURS) + 1)

    # ----------------------------------------------
    # 2.1 库存状态变量
    # ----------------------------------------------

    stock = model.addVars(
        ids, states,
        lb=0,
        name="stock"
    )

    # 调拨完成后库存
    pre = model.addVars(
        ids, steps,
        lb=0,
        name="after_transfer"
    )

    # 借车完成后库存
    post_dep = model.addVars(
        ids, steps,
        lb=0,
        name="after_departures"
    )

    # 借车阶段按需进行外部补车
    external_in = model.addVars(
        ids, steps,
        lb=0,
        name="external_in"
    )

    # 归还阶段进行外部疏导
    external_out = model.addVars(
        ids, steps,
        lb=0,
        name="external_out"
    )

    for i in ids:

        for k in states:
            stock[i, k].UB = capacity[i]

        for k in steps:
            pre[i, k].UB = capacity[i]
            post_dep[i, k].UB = capacity[i]

        model.addConstr(
            stock[i, 0] == initial[i],
            name=f"initial_{i}"
        )

    # ----------------------------------------------
    # 2.2 候选调拨关系
    # ----------------------------------------------

    outs = {i: [] for i in ids}
    ins = {i: [] for i in ids}

    for i, j in edges:
        outs[i].append(j)
        ins[j].append(i)

    if with_moves:

        indices = [
            (i, j, k)
            for i, j in edges
            for k in steps
        ]

        move = model.addVars(
            indices,
            vtype=GRB.INTEGER,
            lb=0,
            name="move"
        )

        activate = model.addVars(
            indices,
            vtype=GRB.BINARY,
            name="activate"
        )

        for i, j, k in indices:

            M = min(
                capacity[i],
                capacity[j],
                MAX_MOVES
            )

            model.addConstr(
                move[i, j, k]
                <= M * activate[i, j, k]
            )

            model.addConstr(
                move[i, j, k]
                >= activate[i, j, k]
            )

        model.addConstr(
            gp.quicksum(
                move[i, j, k]
                for i, j, k in indices
            ) <= MAX_MOVES,
            name="move_budget"
        )

    else:

        indices = []
        move = None
        activate = None

    # ----------------------------------------------
    # 2.3 分阶段库存约束
    # ----------------------------------------------

    for k, hour in enumerate(HOURS):

        for i in ids:

            if with_moves:

                send = gp.quicksum(
                    move[i, j, k]
                    for j in outs[i]
                )

                recv = gp.quicksum(
                    move[j, i, k]
                    for j in ins[i]
                )

                # 只能调出小时初已经拥有的车辆
                model.addConstr(
                    send <= stock[i, k]
                )

                # 调入不能超过小时初空位
                model.addConstr(
                    recv <= capacity[i] - stock[i, k]
                )

            else:

                send = 0
                recv = 0

            # 阶段1：完成站间调拨
            model.addConstr(
                pre[i, k]
                == stock[i, k] + recv - send,
                name=f"transfer_stage_{i}_{k}"
            )

            # 阶段2：完成全部借车
            # 注意：此时尚未使用本小时归还量
            model.addConstr(
                post_dep[i, k]
                == pre[i, k]
                + external_in[i, k]
                - dep[i, hour],
                name=f"departure_stage_{i}_{k}"
            )

            # 阶段3：处理归还及必要的外部疏导
            model.addConstr(
                stock[i, k + 1]
                == post_dep[i, k]
                + arr[i, hour]
                - external_out[i, k],
                name=f"arrival_stage_{i}_{k}"
            )

    # ----------------------------------------------
    # 2.4 目标函数
    # ----------------------------------------------

    if with_moves:

        transport_expr = gp.quicksum(
            DISTANCE_COST
            * distance[i, j]
            * move[i, j, k]
            for i, j, k in indices
        )

        fixed_expr = FIXED_COST * gp.quicksum(
            activate[i, j, k]
            for i, j, k in indices
        )

    else:

        transport_expr = 0
        fixed_expr = 0

    external_in_expr = EXTRA_COST * gp.quicksum(
        external_in[i, k]
        for i in ids
        for k in steps
    )

    external_out_expr = REMOVE_COST * gp.quicksum(
        external_out[i, k]
        for i in ids
        for k in steps
    )

    model.setObjective(
        transport_expr
        + fixed_expr
        + external_in_expr
        + external_out_expr,
        GRB.MINIMIZE
    )

    # ----------------------------------------------
    # 2.5 求解
    # ----------------------------------------------

    model.optimize()

    if model.SolCount == 0:

        raise RuntimeError(
            f"{mode} 无可行解，"
            f"Gurobi Status={model.Status}"
        )

    if model.Status == GRB.OPTIMAL:

        status = "OPTIMAL_WITH_TOLERANCE"

    elif model.Status == GRB.TIME_LIMIT:

        status = "TIME_LIMIT_WITH_INCUMBENT"

    else:

        status = (
            f"STATUS_{model.Status}_WITH_INCUMBENT"
        )

    # ----------------------------------------------
    # 2.6 提取调拨方案
    # ----------------------------------------------

    transfers = []
    station_rows = []
    by_hour = []

    for k, hour in enumerate(HOURS):

        moves_hour = []

        if with_moves:

            for i, j in edges:

                qty = move[i, j, k].X

                if qty <= TOL:
                    continue

                if abs(qty - round(qty)) > TOL:

                    raise AssertionError(
                        "调拨出现非整数车辆"
                    )

                if activate[i, j, k].X < 1 - TOL:

                    raise AssertionError(
                        "存在未启用的调拨关系"
                    )

                moves_hour.append({
                    "model": mode,
                    "hour": hour,
                    "origin_id": i,
                    "destination_id": j,
                    "bikes": int(round(qty)),
                    "distance_km": round(
                        distance[i, j], 6
                    ),
                    "transport_cost": round(
                        qty * DISTANCE_COST
                        * distance[i, j], 6
                    )
                })

        transfers.extend(moves_hour)

        # ------------------------------------------
        # 2.7 逐站、逐小时审计
        # ------------------------------------------

        for i in ids:

            # 不进行外部补车时，借车阶段的缺口
            shortage_risk = max(
                0.0,
                dep[i, hour] - pre[i, k].X
            )

            # 允许应急补车后的剩余缺口
            protected_gap = max(
                0.0,
                dep[i, hour]
                - pre[i, k].X
                - external_in[i, k].X
            )

            # 调拨阶段库存守恒残差
            res_pre = (
                pre[i, k].X
                - stock[i, k].X
            )

            if with_moves:

                res_pre -= sum(
                    move[j, i, k].X
                    for j in ins[i]
                )

                res_pre += sum(
                    move[i, j, k].X
                    for j in outs[i]
                )

            # 借车阶段库存守恒残差
            res_dep = (
                post_dep[i, k].X
                - pre[i, k].X
                - external_in[i, k].X
                + dep[i, hour]
            )

            # 归还阶段库存守恒残差
            res_ret = (
                stock[i, k + 1].X
                - post_dep[i, k].X
                - arr[i, hour]
                + external_out[i, k].X
            )

            if max(
                abs(res_pre),
                abs(res_dep),
                abs(res_ret)
            ) > TOL:

                raise AssertionError(
                    f"站点 {i} 小时 {hour} "
                    "分阶段库存不守恒"
                )

            if protected_gap > TOL:

                raise AssertionError(
                    f"站点 {i} 小时 {hour} "
                    "借车阶段供给不足"
                )

            station_rows.append({

                "model": mode,
                "hour": hour,
                "station_id": i,

                "stock_start": stock[i, k].X,

                "stock_after_transfer":
                    pre[i, k].X,

                "observed_departures":
                    dep[i, hour],

                "external_in_during_departures":
                    external_in[i, k].X,

                "stock_after_departures":
                    post_dep[i, k].X,

                "observed_arrivals":
                    arr[i, hour],

                "external_out_during_arrivals":
                    external_out[i, k].X,

                "stock_end":
                    stock[i, k + 1].X,

                "gap_if_no_external_in":
                    shortage_risk,

                "gap_after_external_in":
                    protected_gap
            })

        # 当前小时25个站点的记录
        subset = station_rows[-len(ids):]

        by_hour.append({

            "model": mode,
            "hour": hour,

            "observed_departures": sum(
                dep[i, hour] for i in ids
            ),

            "observed_arrivals": sum(
                arr[i, hour] for i in ids
            ),

            "external_in": sum(
                r["external_in_during_departures"]
                for r in subset
            ),

            "external_out": sum(
                r["external_out_during_arrivals"]
                for r in subset
            ),

            "moved_bikes": sum(
                r["bikes"] for r in moves_hour
            ),

            "active_arc_hours": len(moves_hour),

            "stock_begin": sum(
                stock[i, k].X for i in ids
            ),

            "stock_end": sum(
                stock[i, k + 1].X for i in ids
            ),

            "gap_if_no_external_in": sum(
                r["gap_if_no_external_in"]
                for r in subset
            ),

            "gap_after_external_in": sum(
                r["gap_after_external_in"]
                for r in subset
            )
        })

    # ----------------------------------------------
    # 2.8 总量及目标函数验收
    # ----------------------------------------------

    total_extra = sum(
        r["external_in"] for r in by_hour
    )

    total_remove = sum(
        r["external_out"] for r in by_hour
    )

    total_moves = sum(
        r["moved_bikes"] for r in by_hour
    )

    if with_moves:

        transport = sum(
            DISTANCE_COST
            * distance[i, j]
            * move[i, j, k].X
            for i, j, k in indices
        )

    else:

        transport = 0.0

    route_fixed = FIXED_COST * len(transfers)

    objective_check = (
        transport
        + route_fixed
        + EXTRA_COST * total_extra
        + REMOVE_COST * total_remove
    )

    if abs(
        objective_check - model.ObjVal
    ) > 1e-4:

        raise AssertionError(
            "目标函数成本分解失败"
        )

    if total_moves > MAX_MOVES + TOL:

        raise AssertionError(
            "总调拨量超过预算"
        )

    first_stock = sum(initial.values())

    last_stock = sum(
        stock[i, len(HOURS)].X
        for i in ids
    )

    net_rides = sum(
        arr[i, t] - dep[i, t]
        for i in ids
        for t in HOURS
    )

    conservation_error = (
        last_stock
        - first_stock
        - net_rides
        - total_extra
        + total_remove
    )

    if abs(conservation_error) > TOL:

        raise AssertionError(
            "区域车辆总量不守恒"
        )

    summary = {

        "model": mode,
        "status": status,

        "objective": round(
            model.ObjVal, 6
        ),

        "transport_cost": round(
            transport, 6
        ),

        "fixed_task_cost": round(
            route_fixed, 6
        ),

        "external_in": round(
            total_extra, 6
        ),

        "external_out": round(
            total_remove, 6
        ),

        "moved_bikes": total_moves,

        "active_arc_hours": len(transfers),

        "stock_start": first_stock,

        "stock_end": round(
            last_stock, 6
        ),

        "max_gap_after_external_in": round(
            max(
                r["gap_after_external_in"]
                for r in station_rows
            ),
            8
        ),

        "best_bound": round(
            model.ObjBound, 6
        ),

        "mip_gap": (
            round(model.MIPGap, 8)
            if with_moves else 0.0
        ),

        "runtime_seconds": round(
            model.Runtime, 4
        )
    }

    return (
        summary,
        by_hour,
        station_rows,
        transfers
    )


# ==================================================
# 3. 保存 CSV
# ==================================================

def save_csv(path, rows, columns=None):

    if not rows and columns is None:
        raise ValueError(
            "空CSV必须指定列名"
        )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=(
                columns
                if columns is not None
                else list(rows[0].keys())
            )
        )

        writer.writeheader()
        writer.writerows(rows)


# ==================================================
# 4. 主程序
# ==================================================

def main():

    print("=" * 60)
    print("Day 4-D：先借后还的多周期 MILP")
    print("=" * 60)

    # 沿用已验收的25站真实数据输入
    data = previous.load_data()

    print()
    print(
        "[2/4] 求解分阶段无调拨基线..."
    )

    baseline = solve(
        data,
        with_moves=False
    )

    print(
        "[3/4] 求解分阶段站间调拨 MILP..."
    )

    optimized = solve(
        data,
        with_moves=True
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    save_csv(
        OUTPUT / "strict_order_comparison.csv",
        [
            baseline[0],
            optimized[0]
        ]
    )

    save_csv(
        OUTPUT / "strict_order_by_hour.csv",
        baseline[1] + optimized[1]
    )

    save_csv(
        OUTPUT / "strict_order_station_detail.csv",
        baseline[2] + optimized[2]
    )

    save_csv(
        OUTPUT / "strict_order_transfer_plan.csv",
        optimized[3],
        [
            "model",
            "hour",
            "origin_id",
            "destination_id",
            "bikes",
            "distance_km",
            "transport_cost"
        ]
    )

    audit = {

        "hours": HOURS,

        "inventory_scenario":
            previous.SCENARIO,

        "station_seed":
            previous.SEED,

        "transfer_budget":
            MAX_MOVES,

        "fixed_cost":
            FIXED_COST,

        "external_in_cost":
            EXTRA_COST,

        "external_out_cost":
            REMOVE_COST,

        "phase_order": [
            "transfer",
            "departures_with_on_demand_external_supply",
            "arrivals_with_external_evacuation"
        ],

        "results": [
            baseline[0],
            optimized[0]
        ],

        "limitations": [
            "观测借还量是必须完成的固定流量目标。",
            "小时平均流量与外部干预变量允许小数。",
            "外部补车在借车阶段可按需即时进行。",
            "外部疏导在归还阶段可同步完成。",
            "尚无订单级OD配对和实际骑行时间。",
            "未考虑调拨货车容量及运输时间。",
            "站点容量采用非同期GBFS代理值。",
            "本模型不是订单级事件仿真。"
        ]
    }

    (
        OUTPUT / "strict_order_audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print(
        "[4/4] 结果保存与分阶段守恒验收完成"
    )

    print()
    print("=" * 60)
    print("Day 4-D 先借后还多周期 MILP 结果")
    print("=" * 60)

    for summary in [
        baseline[0],
        optimized[0]
    ]:

        print(
            f"{summary['model']}: "
            f"{summary['status']}"
        )

        print(
            f"  目标值="
            f"{summary['objective']:.4f}"
        )

        print(
            f"  外部补车="
            f"{summary['external_in']:.3f}"
        )

        print(
            f"  外部疏导="
            f"{summary['external_out']:.3f}"
        )

        print(
            f"  站间调拨="
            f"{summary['moved_bikes']}"
        )

        print(
            f"  调拨关系-小时数="
            f"{summary['active_arc_hours']}"
        )

        print(
            f"  MIP Gap="
            f"{summary['mip_gap']:.4%}"
        )

        print(
            f"  借车阶段剩余缺口="
            f"{summary['max_gap_after_external_in']:.6f}"
        )

    print()
    print("输出目录：", OUTPUT)


if __name__ == "__main__":
    main()
