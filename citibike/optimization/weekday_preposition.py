
# -*- coding: utf-8 -*-
"""
Day 6-A:
仅使用训练期流量优化午夜预调拨，
随后在开发日进行30分钟超时事件仿真。

直接运行，不需要命令行参数。
"""

from pathlib import Path
import json
import math

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

import citibike.simulation.event_baseline as event_base
import citibike.simulation.timeout_simulation as timeout_replay


from citibike.paths import ROOT

OUTPUT = (
    ROOT / "02_RESULTS" / "DAY6_PREPOSITION"
)

PROFILE_FILE = (
    ROOT / "02_RESULTS"
    / "DAY4_HOURLY_PREP"
    / "train_hourly_station_profiles.csv"
)

SEED = "5980.10"
SCENARIO = "balanced_50"

# 训练模型使用前12小时；
# 开发日事件回放仍覆盖完整24小时。
HOURS = list(range(12))

NEIGHBORS = 4
MOVE_BUDGET = 80

FIXED_COST = 3.0
COST_PER_BIKE_KM = 1.0

VIRTUAL_SUPPLY_COST = 20.0
VIRTUAL_EVAC_COST = 20.0

TIME_LIMIT = 90
MIP_GAP = 0.001
TOL = 1e-5


# ==================================================
# 1. 距离
# ==================================================

def distance_km(p, q):

    a, b = map(math.radians, p)
    c, d = map(math.radians, q)

    v = (
        math.sin((c - a) / 2) ** 2
        + math.cos(a)
        * math.cos(c)
        * math.sin((d - b) / 2) ** 2
    )

    return (
        2 * 6371.0088
        * math.asin(
            math.sqrt(
                max(0.0, min(1.0, v))
            )
        )
    )


# ==================================================
# 2. 只加载训练输入
# ==================================================

def load_training_data():

    print(
        "[1/4] 加载训练期小时流量及冻结库存..."
    )

    for file in [
        event_base.STATIONS_FILE,
        event_base.INVENTORY_FILE,
        PROFILE_FILE
    ]:

        if not file.is_file():
            raise FileNotFoundError(file)

    stations = pd.read_csv(
        event_base.STATIONS_FILE,
        dtype={
            "station_id": str,
            "candidate_seed": str
        }
    )

    stations = stations.loc[
        stations["candidate_seed"] == SEED
    ].copy()

    inventory = pd.read_csv(
        event_base.INVENTORY_FILE,
        dtype={"station_id": str}
    )

    inventory = inventory.loc[
        inventory["scenario"] == SCENARIO
    ].copy()

    profile = pd.read_csv(
        PROFILE_FILE,
        dtype={"station_id": str}
    )

    profile = profile.loc[
        profile["hour"].isin(HOURS)
    ].copy()

    ids = sorted(
        stations["station_id"].tolist()
    )

    if len(ids) != 25 or len(set(ids)) != 25:
        raise ValueError(
            "主区域不是25个不同站点"
        )

    if (
        len(inventory) != 25
        or inventory["station_id"].duplicated().any()
        or set(inventory["station_id"]) != set(ids)
    ):
        raise ValueError(
            "初始库存与主区域不匹配"
        )

    if (
        len(profile) != 25 * len(HOURS)
        or profile.duplicated(
            ["station_id", "hour"]
        ).any()
        or set(profile["station_id"]) != set(ids)
    ):
        raise ValueError(
            "训练期逐小时数据不完整"
        )

    stations = stations.set_index("station_id")
    inventory = inventory.set_index("station_id")

    profile = profile.set_index(
        ["station_id", "hour"]
    )

    capacity = {}
    initial = {}
    locations = {}

    departures = {}
    arrivals = {}

    for i in ids:

        cap = float(
            stations.at[i, "proxy_capacity"]
        )

        stock = float(
            inventory.at[i, "initial_inventory"]
        )

        if not (
            math.isfinite(cap)
            and math.isfinite(stock)
            and cap.is_integer()
            and stock.is_integer()
            and 0 <= stock <= cap
            and cap > 0
        ):
            raise ValueError(
                f"站点{i}库存或容量非法"
            )

        capacity[i] = int(cap)
        initial[i] = int(stock)

        locations[i] = (
            float(stations.at[i, "latitude"]),
            float(stations.at[i, "longitude"])
        )

        for hour in HOURS:

            dep = float(
                profile.at[
                    (i, hour), "avg_departures"
                ]
            )

            arr = float(
                profile.at[
                    (i, hour), "avg_arrivals"
                ]
            )

            if not (
                math.isfinite(dep)
                and math.isfinite(arr)
                and dep >= 0
                and arr >= 0
            ):
                raise ValueError(
                    f"站点{i}小时{hour}流量异常"
                )

            departures[i, hour] = dep
            arrivals[i, hour] = arr

    # 每站四条最近邻有向边
    arcs = []
    distance = {}

    for i in ids:

        nearby = sorted(
            [
                j for j in ids
                if j != i
            ],
            key=lambda j: (
                distance_km(
                    locations[i],
                    locations[j]
                ),
                j
            )
        )[:NEIGHBORS]

        for j in nearby:

            arcs.append((i, j))

            distance[i, j] = distance_km(
                locations[i],
                locations[j]
            )

    print(
        f"站点={len(ids)}，"
        f"初始车辆={sum(initial.values())}，"
        f"容量={sum(capacity.values())}"
    )

    print(
        f"训练小时数={len(HOURS)}，"
        f"候选有向关系={len(arcs)}"
    )

    return (
        ids,
        capacity,
        initial,
        departures,
        arrivals,
        arcs,
        distance
    )


# ==================================================
# 3. Gurobi 预调拨优化
# ==================================================

def optimize(data, allow_moves):

    (
        ids,
        capacity,
        initial,
        departures,
        arrivals,
        arcs,
        distance
    ) = data

    name = (
        "preposition"
        if allow_moves
        else "baseline"
    )

    model = gp.Model(name)

    model.Params.OutputFlag = 0
    model.Params.TimeLimit = TIME_LIMIT

    if allow_moves:
        model.Params.MIPGap = MIP_GAP

    T = len(HOURS)

    # 各小时库存
    I = model.addVars(
        ids,
        range(T + 1),
        lb=0,
        name="inventory"
    )

    # 借车完成后的库存
    H = model.addVars(
        ids,
        range(T),
        lb=0,
        name="after_departures"
    )

    # 规划阶段虚拟外部干预
    S = model.addVars(
        ids,
        range(T),
        lb=0,
        name="virtual_supply"
    )

    E = model.addVars(
        ids,
        range(T),
        lb=0,
        name="virtual_evacuation"
    )

    for i in ids:

        for t in range(T + 1):
            I[i, t].UB = capacity[i]

        for t in range(T):
            H[i, t].UB = capacity[i]

    inbound = {
        i: [
            j for j in ids
            if (j, i) in distance
        ]
        for i in ids
    }

    outbound = {
        i: [
            j for j in ids
            if (i, j) in distance
        ]
        for i in ids
    }

    # ----------------------------------------------
    # 3.1 午夜一次性预调拨
    # ----------------------------------------------

    if allow_moves:

        x = model.addVars(
            arcs,
            lb=0,
            vtype=GRB.INTEGER,
            name="midnight_transfer"
        )

        y = model.addVars(
            arcs,
            vtype=GRB.BINARY,
            name="activate"
        )

        for i, j in arcs:

            M = min(
                initial[i],
                capacity[j] - initial[j],
                MOVE_BUDGET
            )

            model.addConstr(
                x[i, j] <= M * y[i, j]
            )

            model.addConstr(
                x[i, j] >= y[i, j]
            )

        model.addConstr(
            gp.quicksum(
                x[i, j]
                for i, j in arcs
            ) <= MOVE_BUDGET,
            name="move_budget"
        )

        for i in ids:

            outgoing = gp.quicksum(
                x[i, j]
                for j in outbound[i]
            )

            incoming = gp.quicksum(
                x[j, i]
                for j in inbound[i]
            )

            # 只能调出原本拥有的车辆
            model.addConstr(
                outgoing <= initial[i]
            )

            # 只能调入原本空闲的车位
            model.addConstr(
                incoming
                <= capacity[i] - initial[i]
            )

            model.addConstr(
                I[i, 0]
                == initial[i]
                + incoming
                - outgoing
            )

        transport_cost = gp.quicksum(
            distance[i, j]
            * COST_PER_BIKE_KM
            * x[i, j]
            for i, j in arcs
        )

        fixed_cost = FIXED_COST * gp.quicksum(
            y[i, j]
            for i, j in arcs
        )

    else:

        x = None

        for i in ids:
            model.addConstr(
                I[i, 0] == initial[i]
            )

        transport_cost = 0
        fixed_cost = 0

    # ----------------------------------------------
    # 3.2 训练期多小时库存递推
    # ----------------------------------------------

    for t, hour in enumerate(HOURS):

        for i in ids:

            # 保守假设：先借后还。
            # 虚拟补车量仅用于训练目标的软约束。
            model.addConstr(
                H[i, t]
                ==
                I[i, t]
                + S[i, t]
                - departures[i, hour]
            )

            model.addConstr(
                I[i, t + 1]
                ==
                H[i, t]
                + arrivals[i, hour]
                - E[i, t]
            )

    # ----------------------------------------------
    # 3.3 训练目标
    # ----------------------------------------------

    model.setObjective(
        transport_cost
        + fixed_cost
        + VIRTUAL_SUPPLY_COST
        * gp.quicksum(
            S[i, t]
            for i in ids
            for t in range(T)
        )
        + VIRTUAL_EVAC_COST
        * gp.quicksum(
            E[i, t]
            for i in ids
            for t in range(T)
        ),
        GRB.MINIMIZE
    )

    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError(
            f"{name}未找到可行解，"
            f"Status={model.Status}"
        )

    # ----------------------------------------------
    # 3.4 提取调拨结果
    # ----------------------------------------------

    moves = []
    new_inventory = dict(initial)

    if allow_moves:

        for i, j in arcs:

            qty = x[i, j].X

            if qty <= TOL:
                continue

            if abs(qty - round(qty)) > TOL:
                raise AssertionError(
                    "调拨数量不是整数"
                )

            qty = int(round(qty))

            moves.append({
                "origin_id": i,
                "destination_id": j,
                "bikes": qty,
                "distance_km": round(
                    distance[i, j], 6
                ),
                "transport_cost": round(
                    qty
                    * distance[i, j]
                    * COST_PER_BIKE_KM,
                    6
                )
            })

            new_inventory[i] -= qty
            new_inventory[j] += qty

    if sum(new_inventory.values()) != sum(
        initial.values()
    ):
        raise AssertionError(
            "预调拨车辆总量不守恒"
        )

    for i in ids:

        if not (
            0 <= new_inventory[i] <= capacity[i]
        ):
            raise AssertionError(
                f"预调拨后库存越界：{i}"
            )

        if abs(
            new_inventory[i] - I[i, 0].X
        ) > TOL:
            raise AssertionError(
                "调拨方案与优化库存不一致"
            )

    # ----------------------------------------------
    # 3.5 核算目标值
    # ----------------------------------------------

    supply = sum(
        S[i, t].X
        for i in ids
        for t in range(T)
    )

    evacuation = sum(
        E[i, t].X
        for i in ids
        for t in range(T)
    )

    move_cost = sum(
        row["bikes"]
        * distance[
            row["origin_id"],
            row["destination_id"]
        ]
        * COST_PER_BIKE_KM
        for row in moves
    )

    task_cost = FIXED_COST * len(moves)

    reconstructed = (
        move_cost
        + task_cost
        + VIRTUAL_SUPPLY_COST * supply
        + VIRTUAL_EVAC_COST * evacuation
    )

    if abs(
        reconstructed - model.ObjVal
    ) > 1e-4:
        raise AssertionError(
            "训练目标值分解错误"
        )

    summary = {
        "model": name,
        "solver_status": (
            "OPTIMAL_WITH_TOLERANCE"
            if model.Status == GRB.OPTIMAL
            else f"STATUS_{model.Status}_WITH_INCUMBENT"
        ),
        "surrogate_objective": round(
            model.ObjVal, 6
        ),
        "virtual_supply": round(
            supply, 6
        ),
        "virtual_evacuation": round(
            evacuation, 6
        ),
        "prepositioned_bikes": sum(
            r["bikes"] for r in moves
        ),
        "active_directed_pairs": len(moves),
        "transport_cost": round(
            move_cost, 6
        ),
        "fixed_cost": round(
            task_cost, 6
        ),
        "mip_gap": round(
            model.MIPGap if allow_moves else 0,
            7
        ),
        "runtime_sec": round(
            model.Runtime, 4
        )
    }

    return summary, moves, new_inventory


# ==================================================
# 4. 训练与开发日事件回放
# ==================================================

def main():

    print("=" * 64)
    print(
        "Day 6-A：训练期午夜预调拨 + 开发日事件验证"
    )
    print("=" * 64)

    training = load_training_data()

    print()
    print(
        "[2/4] Gurobi求解训练期调拨规划..."
    )

    baseline_model, _, stock0 = optimize(
        training,
        allow_moves=False
    )

    plan_model, moves, newstock = optimize(
        training,
        allow_moves=True
    )

    print(
        "训练代理目标值："
        f"{baseline_model['surrogate_objective']:.2f}"
        " -> "
        f"{plan_model['surrogate_objective']:.2f}"
    )

    print(
        f"预调拨={plan_model['prepositioned_bikes']}辆，"
        f"启用关系={plan_model['active_directed_pairs']}"
    )

    print(
        f"MIP Gap={plan_model['mip_gap']:.4%}"
    )

    # ----------------------------------------------
    # 4.1 读取开发日真实订单
    # ----------------------------------------------

    print()
    print(
        "[3/4] 使用相同30分钟政策进行开发日回放..."
    )

    (
        ids,
        capacity,
        initial,
        trips,
        expected_starts,
        expected_ends,
        _
    ) = event_base.read_inputs()

    if (
        ids != training[0]
        or capacity != training[1]
        or initial[SCENARIO] != stock0
    ):
        raise AssertionError(
            "规划模型与事件仿真输入不一致"
        )

    # 原始balanced_50
    baseline, base_waits, base_hours = (
        timeout_replay.replay(
            ids,
            capacity,
            stock0,
            trips,
            "baseline_balanced_50"
        )
    )

    # 使用训练期生成的新库存
    candidate, plan_waits, plan_hours = (
        timeout_replay.replay(
            ids,
            capacity,
            newstock,
            trips,
            "training_preposition"
        )
    )

    # ----------------------------------------------
    # 4.2 开发日数据与车辆账本验收
    # ----------------------------------------------

    for result in [baseline, candidate]:

        if result["requests"] != expected_starts:
            raise AssertionError(
                "借车请求数量不一致"
            )

        if (
            result["arrival_attempts"]
            + result[
                "cancelled_internal_arrivals_today"
            ]
            != expected_ends
        ):
            raise AssertionError(
                "归还订单对账失败"
            )

        if result["vehicle_conservation"] != "PASS":
            raise AssertionError(
                "车辆守恒未通过"
            )

    # 与 Day 5-E 冻结的balanced_50比较
    frozen_file = (
        timeout_replay.OUT_DIR
        / "timeout_summary.csv"
    )

    if not frozen_file.is_file():
        raise FileNotFoundError(frozen_file)

    frozen = pd.read_csv(
        frozen_file
    ).set_index("scenario").loc[SCENARIO]

    for metric in [
        "requests",
        "borrow_rejected",
        "wait_occurrences",
        "timeout_external_exit",
        "pending_at_midnight"
    ]:

        if int(baseline[metric]) != int(
            frozen[metric]
        ):
            raise AssertionError(
                f"无调拨冻结基线不一致：{metric}"
            )

    print()
    print(
        "无调拨基线："
        f"借车拒绝={baseline['borrow_rejected']}，"
        f"归还超时={baseline['timeout_external_exit']}，"
        f"目标站归还率="
        f"{baseline['target_station_docking_rate']:.2%}"
    )

    print(
        "训练期预调拨："
        f"借车拒绝={candidate['borrow_rejected']}，"
        f"归还超时={candidate['timeout_external_exit']}，"
        f"目标站归还率="
        f"{candidate['target_station_docking_rate']:.2%}"
    )

    # ----------------------------------------------
    # 4.3 保存结果
    # ----------------------------------------------

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(
        [baseline_model, plan_model]
    ).to_csv(
        OUTPUT / "training_model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        moves,
        columns=[
            "origin_id",
            "destination_id",
            "bikes",
            "distance_km",
            "transport_cost"
        ]
    ).to_csv(
        OUTPUT / "training_preposition_plan.csv",
        index=False,
        encoding="utf-8-sig"
    )

    inventory_rows = []

    for i in ids:

        inventory_rows.append({
            "station_id": i,
            "capacity": capacity[i],
            "before": stock0[i],
            "after": newstock[i],
            "delta": newstock[i] - stock0[i]
        })

    pd.DataFrame(inventory_rows).to_csv(
        OUTPUT / "initial_inventory_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        [baseline, candidate]
    ).to_csv(
        OUTPUT / "devday_service_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        base_hours + plan_hours
    ).to_csv(
        OUTPUT / "devday_station_hour.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        base_waits + plan_waits
    ).to_csv(
        OUTPUT / "devday_wait_log.csv",
        index=False,
        encoding="utf-8-sig"
    )

    audit = {
        "training_dates": (
            "2026-04-01 to 2026-04-20, weekdays"
        ),
        "training_hours": "00:00-11:59",
        "development_date": event_base.DATE,
        "starting_scenario": SCENARIO,
        "preposition_timing": (
            "Before test-day midnight; "
            "instantaneous relocation approximation"
        ),
        "fleet_before_and_after": sum(
            stock0.values()
        ),
        "move_budget": MOVE_BUDGET,
        "wait_limit_minutes":
            timeout_replay.WAIT_LIMIT_MIN,
        "training_objective_is_not_actual_service_cost":
            True,
        "evaluation_status": (
            "Development diagnostic only; "
            "final untouched dates remain unused"
        ),
        "limitations": [
            "训练输入使用历史已完成骑行的小时平均值。",
            "训练期虚拟补车和疏导为优化松弛变量。",
            "预调拨尚未考虑卡车容量和运输时间。",
            "30分钟超时后车辆假设离开研究区域。",
            "4月21日已被用于开发诊断，不是最终盲测。",
            "训练目标值与事件级服务损失不是同一指标。"
        ]
    }

    (
        OUTPUT / "audit.json"
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
        "[4/4] 规划、测试和车辆守恒验收完成"
    )

    print("输出目录：", OUTPUT)


if __name__ == "__main__":
    main()
