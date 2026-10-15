
# -*- coding: utf-8 -*-
"""
Day 4-B: Citi Bike 四小时多周期库存优化 MILP

比较：
1. 不允许站间调拨
2. 允许站间调拨，四小时合计最多80辆

模型性质：
- 25个真实站点
- 合成初始库存
- 历史已完成骑行的小时平均借还流量
- 整数调拨 + 二元任务启用变量
- 小时级库存流体近似
- 外部应急补车和疏导变量

直接运行，不需要命令行参数。
"""

from pathlib import Path
import csv
import json
import math

import pandas as pd
import gurobipy as gp
from gurobipy import GRB


# ============================================================
# 1. 实验参数
# ============================================================

from project_paths import ROOT
BASE = ROOT / "02_RESULTS"

SEED = "5980.10"
SCENARIO = "balanced_50"

HOURS = [15, 16, 17, 18]

# 每站只连接4个最近邻
NEIGHBORS = 4

# 四小时总调拨工作量上限
MOVE_BUDGET = 80

# 假设成本参数
COST_PER_BIKE_KM = 1.0
FIXED_COST_PER_ARC_HOUR = 3.0

EXTRA_COST = 20.0
REMOVE_COST = 20.0

TIME_LIMIT_SECONDS = 90
MIP_GAP = 0.001


STATION_FILE = (
    BASE
    / "DAY2_A_FIXED"
    / "day2_selected_candidates.csv"
)

INVENTORY_FILE = (
    BASE
    / "DAY2_B_INVENTORY"
    / "initial_inventory_scenarios.csv"
)

PROFILE_FILE = (
    BASE
    / "DAY4_HOURLY_PREP"
    / "train_hourly_station_profiles.csv"
)

OUT_DIR = (
    BASE
    / "DAY4_MULTIPERIOD_MILP"
)


# ============================================================
# 2. 经纬度距离
# ============================================================

def km(a, b):

    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)

    term = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin((lon2 - lon1) / 2) ** 2
    )

    term = min(1.0, max(0.0, term))

    return (
        2 * 6371.0088
        * math.asin(math.sqrt(term))
    )


# ============================================================
# 3. 读取数据
# ============================================================

def load_data():

    print(
        "[1/4] 读取25站、初始库存及四小时历史流量..."
    )

    for path in [
        STATION_FILE,
        INVENTORY_FILE,
        PROFILE_FILE
    ]:

        if not path.is_file():
            raise FileNotFoundError(
                f"找不到输入文件：{path}"
            )

    # ----------------------------
    # 3.1 站点
    # ----------------------------

    st = pd.read_csv(
        STATION_FILE,
        dtype={
            "station_id": str,
            "candidate_seed": str
        }
    )

    st = st[
        st["candidate_seed"] == SEED
    ].copy()

    ids = sorted(
        st["station_id"].tolist()
    )

    if len(ids) != 25:
        raise ValueError(
            "主实验区域必须有25个站点"
        )

    if len(set(ids)) != 25:
        raise ValueError(
            "存在重复站点ID"
        )

    # ----------------------------
    # 3.2 初始库存
    # ----------------------------

    inv = pd.read_csv(
        INVENTORY_FILE,
        dtype={"station_id": str}
    )

    inv = inv[
        inv["scenario"] == SCENARIO
    ].copy()

    if inv["station_id"].duplicated().any():
        raise ValueError(
            "初始库存存在重复站点"
        )

    if set(inv["station_id"]) != set(ids):
        raise ValueError(
            "库存站点与主实验区域不一致"
        )

    # ----------------------------
    # 3.3 小时流量
    # ----------------------------

    profile = pd.read_csv(
        PROFILE_FILE,
        dtype={"station_id": str}
    )

    profile = profile[
        profile["hour"].isin(HOURS)
    ].copy()

    if profile.duplicated(
        ["station_id", "hour"]
    ).any():

        raise ValueError(
            "存在重复的站点-小时记录"
        )

    if len(profile) != 25 * len(HOURS):
        raise ValueError(
            "小时数据未完整覆盖25站"
        )

    if set(profile["station_id"]) != set(ids):
        raise ValueError(
            "小时数据站点集合不一致"
        )

    st = st.set_index("station_id")
    inv = inv.set_index("station_id")

    profile = profile.set_index(
        ["station_id", "hour"]
    )

    capacity = {}
    initial = {}
    locations = {}

    departure = {}
    arrival = {}

    # ----------------------------
    # 3.4 参数字典
    # ----------------------------

    for i in ids:

        cap_float = float(
            st.at[i, "proxy_capacity"]
        )

        inv_float = float(
            inv.at[i, "initial_inventory"]
        )

        if (
            not math.isfinite(cap_float)
            or not math.isfinite(inv_float)
            or cap_float <= 0
        ):
            raise ValueError(
                f"站点{i}容量或库存非法"
            )

        if (
            abs(cap_float - round(cap_float)) > 1e-8
            or abs(inv_float - round(inv_float)) > 1e-8
        ):
            raise ValueError(
                f"站点{i}容量或库存不是整数"
            )

        capacity[i] = int(round(cap_float))
        initial[i] = int(round(inv_float))

        if not (
            0 <= initial[i] <= capacity[i]
        ):
            raise ValueError(
                f"站点{i}初始库存超过容量"
            )

        locations[i] = (
            float(st.at[i, "latitude"]),
            float(st.at[i, "longitude"])
        )

        for t in HOURS:

            d = float(
                profile.at[
                    (i, t), "avg_departures"
                ]
            )

            a = float(
                profile.at[
                    (i, t), "avg_arrivals"
                ]
            )

            if not (
                math.isfinite(d)
                and math.isfinite(a)
                and d >= 0
                and a >= 0
            ):
                raise ValueError(
                    f"站点{i}小时{t}流量非法"
                )

            departure[i, t] = d
            arrival[i, t] = a

    # ----------------------------
    # 3.5 构建最近邻调拨网络
    # ----------------------------

    edges = []
    distance = {}

    for i in ids:

        nearby = sorted(
            [
                j for j in ids
                if j != i
            ],
            key=lambda j: (
                km(locations[i], locations[j]),
                j
            )
        )[:NEIGHBORS]

        for j in nearby:

            edges.append((i, j))

            distance[i, j] = km(
                locations[i],
                locations[j]
            )

    print(
        f"站点={len(ids)}；"
        f"容量合计={sum(capacity.values())}；"
        f"初始库存={sum(initial.values())}"
    )

    print(
        f"四小时={HOURS}；"
        f"候选有向关系={len(edges)}"
    )

    for t in HOURS:

        d = sum(
            departure[i, t]
            for i in ids
        )

        a = sum(
            arrival[i, t]
            for i in ids
        )

        print(
            f"{t}:00 借出={d:.2f}，"
            f"归还={a:.2f}，"
            f"净流入={a-d:.2f}"
        )

    return (
        ids,
        capacity,
        initial,
        departure,
        arrival,
        edges,
        distance
    )


# ============================================================
# 4. 构建与求解 Gurobi 模型
# ============================================================

def optimize(data, allow_moves):

    (
        ids,
        capacity,
        initial,
        departure,
        arrival,
        edges,
        distance
    ) = data

    name = (
        "with_rebalancing"
        if allow_moves
        else "without_rebalancing"
    )

    model = gp.Model(name)

    model.Params.OutputFlag = 0
    model.Params.TimeLimit = TIME_LIMIT_SECONDS

    if allow_moves:
        model.Params.MIPGap = MIP_GAP

    steps = range(len(HOURS))

    # ========================================================
    # 4.1 决策变量
    # ========================================================

    # stock[i,k]：
    # 第k个时段开始时的车辆库存
    stock = model.addVars(
        ids,
        range(len(HOURS) + 1),
        lb=0,
        name="inventory"
    )

    # extra[i,k]：
    # 假设的外部应急补车量
    extra = model.addVars(
        ids,
        steps,
        lb=0,
        name="external_in"
    )

    # remove[i,k]：
    # 假设的外部车辆疏导量
    remove = model.addVars(
        ids,
        steps,
        lb=0,
        name="external_out"
    )

    # 站点容量与初始库存
    for i in ids:

        for k in range(len(HOURS) + 1):
            stock[i, k].UB = capacity[i]

        model.addConstr(
            stock[i, 0] == initial[i],
            name=f"initial_{i}"
        )

    # ========================================================
    # 4.2 调拨整数变量与二元变量
    # ========================================================

    x = None
    y = None

    if allow_moves:

        arc_times = [
            (i, j, k)
            for i, j in edges
            for k in steps
        ]

        # 整数调拨数量
        x = model.addVars(
            arc_times,
            lb=0,
            vtype=GRB.INTEGER,
            name="transfer"
        )

        # 调拨关系启用变量
        y = model.addVars(
            arc_times,
            vtype=GRB.BINARY,
            name="activate"
        )

        for i, j, k in arc_times:

            M = min(
                capacity[i],
                capacity[j],
                MOVE_BUDGET
            )

            # 没有启用则不能调拨
            model.addConstr(
                x[i, j, k]
                <= M * y[i, j, k]
            )

            # 启用至少调拨1辆
            model.addConstr(
                x[i, j, k]
                >= y[i, j, k]
            )

        # 四小时总调拨预算
        model.addConstr(
            gp.quicksum(
                x[i, j, k]
                for i, j, k in arc_times
            ) <= MOVE_BUDGET,
            name="four_hour_move_budget"
        )

    # ========================================================
    # 4.3 建立流入与流出邻接表
    # ========================================================

    out_neighbors = {
        i: [
            j for a, j in edges
            if a == i
        ]
        for i in ids
    }

    in_neighbors = {
        i: [
            j for j, a in edges
            if a == i
        ]
        for i in ids
    }

    # ========================================================
    # 4.4 多周期库存递推
    # ========================================================

    for k, hour in enumerate(HOURS):

        for i in ids:

            if allow_moves:

                send = gp.quicksum(
                    x[i, j, k]
                    for j in out_neighbors[i]
                )

                receive = gp.quicksum(
                    x[j, i, k]
                    for j in in_neighbors[i]
                )

                # 调出不能超过期初库存
                model.addConstr(
                    send <= stock[i, k],
                    name=f"send_{i}_{k}"
                )

                # 调入不能超过期初空位
                model.addConstr(
                    receive
                    <= capacity[i] - stock[i, k],
                    name=f"receive_{i}_{k}"
                )

            else:

                send = 0
                receive = 0

            # 核心库存守恒方程
            #
            # 下一时段库存
            # = 当前库存
            # + 站间调入 - 站间调出
            # + 观测归还 - 观测借出
            # + 外部补车 - 外部疏导

            model.addConstr(

                stock[i, k + 1]
                ==
                stock[i, k]
                + receive
                - send
                + arrival[i, hour]
                - departure[i, hour]
                + extra[i, k]
                - remove[i, k],

                name=f"balance_{i}_{k}"
            )

    # ========================================================
    # 4.5 目标函数
    # ========================================================

    if allow_moves:

        transport_cost = gp.quicksum(
            distance[i, j]
            * COST_PER_BIKE_KM
            * x[i, j, k]

            for i, j in edges
            for k in steps
        )

        fixed_cost = (
            FIXED_COST_PER_ARC_HOUR
            * gp.quicksum(
                y[i, j, k]

                for i, j in edges
                for k in steps
            )
        )

    else:

        transport_cost = 0
        fixed_cost = 0

    emergency_cost = (
        EXTRA_COST
        * gp.quicksum(
            extra[i, k]
            for i in ids
            for k in steps
        )
    )

    evacuation_cost = (
        REMOVE_COST
        * gp.quicksum(
            remove[i, k]
            for i in ids
            for k in steps
        )
    )

    model.setObjective(
        transport_cost
        + fixed_cost
        + emergency_cost
        + evacuation_cost,
        GRB.MINIMIZE
    )

    # ========================================================
    # 4.6 求解
    # ========================================================

    model.optimize()

    if model.SolCount == 0:

        raise RuntimeError(
            f"{name} 未找到可行解；"
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

    # ========================================================
    # 4.7 提取调拨方案与库存
    # ========================================================

    transfers = []
    by_hour = []
    station_rows = []

    for k, hour in enumerate(HOURS):

        moved = 0
        routes = 0

        if allow_moves:

            for i, j in edges:

                value = x[i, j, k].X

                if value <= 1e-6:
                    continue

                if abs(
                    value - round(value)
                ) > 1e-5:

                    raise AssertionError(
                        "出现非整数调拨"
                    )

                if y[i, j, k].X < 1 - 1e-5:

                    raise AssertionError(
                        "未启用关系上发生了调拨"
                    )

                bikes = int(round(value))

                moved += bikes
                routes += 1

                transfers.append({
                    "hour": hour,
                    "origin_id": i,
                    "destination_id": j,
                    "bikes": bikes,
                    "distance_km": round(
                        distance[i, j], 6
                    ),
                    "transport_cost": round(
                        bikes
                        * distance[i, j]
                        * COST_PER_BIKE_KM,
                        6
                    )
                })

        # 每个站点每小时状态
        for i in ids:

            station_rows.append({

                "model": name,
                "hour": hour,
                "station_id": i,

                "stock_begin": stock[i, k].X,

                "observed_departures":
                    departure[i, hour],

                "observed_arrivals":
                    arrival[i, hour],

                "external_in":
                    extra[i, k].X,

                "external_out":
                    remove[i, k].X,

                "stock_end":
                    stock[i, k + 1].X
            })

        # 区域每小时汇总
        by_hour.append({

            "model": name,
            "hour": hour,

            "observed_departures": sum(
                departure[i, hour]
                for i in ids
            ),

            "observed_arrivals": sum(
                arrival[i, hour]
                for i in ids
            ),

            "external_in": sum(
                extra[i, k].X
                for i in ids
            ),

            "external_out": sum(
                remove[i, k].X
                for i in ids
            ),

            "moved_bikes": moved,

            "active_arc_hours": routes,

            "stock_begin": sum(
                stock[i, k].X
                for i in ids
            ),

            "stock_end": sum(
                stock[i, k + 1].X
                for i in ids
            )
        })

    # ========================================================
    # 4.8 汇总与守恒验收
    # ========================================================

    total_extra = sum(
        extra[i, k].X
        for i in ids
        for k in steps
    )

    total_remove = sum(
        remove[i, k].X
        for i in ids
        for k in steps
    )

    total_moves = sum(
        r["bikes"]
        for r in transfers
    )

    if total_moves > MOVE_BUDGET + 1e-5:

        raise AssertionError(
            "调拨数量超过预算"
        )

    # 全区域累计观测净流量
    net_observed = sum(
        arrival[i, hour] - departure[i, hour]

        for i in ids
        for hour in HOURS
    )

    ending = sum(
        stock[i, len(HOURS)].X
        for i in ids
    )

    # 全区域车辆守恒
    difference = (
        ending
        - sum(initial.values())
        - net_observed
        - total_extra
        + total_remove
    )

    if abs(difference) > 1e-5:

        raise AssertionError(
            "跨时段车辆守恒失败"
        )

    # 每个站点每个小时的守恒检查
    for k in steps:

        for i in ids:

            if allow_moves:

                inflow = sum(
                    x[j, i, k].X
                    for j in in_neighbors[i]
                )

                outflow = sum(
                    x[i, j, k].X
                    for j in out_neighbors[i]
                )

            else:

                inflow = 0
                outflow = 0

            residual = (

                stock[i, k + 1].X
                - stock[i, k].X

                - inflow
                + outflow

                - arrival[i, HOURS[k]]
                + departure[i, HOURS[k]]

                - extra[i, k].X
                + remove[i, k].X
            )

            if abs(residual) > 1e-5:

                raise AssertionError(
                    f"站点{i}小时{HOURS[k]}不守恒"
                )

    # ========================================================
    # 4.9 目标函数分解验收
    # ========================================================

    move_cost = sum(

        r["bikes"]
        * distance[
            r["origin_id"],
            r["destination_id"]
        ]
        * COST_PER_BIKE_KM

        for r in transfers
    )

    task_cost = (
        len(transfers)
        * FIXED_COST_PER_ARC_HOUR
    )

    calculated_objective = (

        move_cost
        + task_cost
        + EXTRA_COST * total_extra
        + REMOVE_COST * total_remove
    )

    if abs(
        calculated_objective - model.ObjVal
    ) > 1e-4:

        raise AssertionError(
            "目标函数成本分解不一致"
        )

    # ========================================================
    # 4.10 模型汇总
    # ========================================================

    summary = {

        "model": name,
        "status": status,

        "objective": model.ObjVal,

        "transport_cost": move_cost,
        "fixed_cost": task_cost,

        "external_in": total_extra,
        "external_out": total_remove,

        "moved_bikes": total_moves,

        "active_arc_hours": len(transfers),

        "initial_stock": sum(initial.values()),
        "final_stock": ending,

        "best_bound": model.ObjBound,

        "mip_gap": (
            model.MIPGap
            if allow_moves
            else 0.0
        ),

        "runtime_seconds": model.Runtime,
        "variables": model.NumVars,
        "constraints": model.NumConstrs
    }

    return (
        summary,
        by_hour,
        transfers,
        station_rows
    )


# ============================================================
# 5. CSV保存函数
# ============================================================

def save_csv(path, rows, columns=None):

    if not rows and columns is None:
        return

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                columns
                if columns is not None
                else list(rows[0].keys())
            )
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 6. 主程序
# ============================================================

def main():

    data = load_data()

    # 无调拨基线
    print()
    print(
        "[2/4] 求解无站间调拨基线..."
    )

    baseline = optimize(
        data,
        allow_moves=False
    )

    # 多周期MILP
    print()
    print(
        "[3/4] 求解四小时多周期 MILP..."
    )

    planned = optimize(
        data,
        allow_moves=True
    )

    # ----------------------------
    # 保存结果
    # ----------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_csv(
        OUT_DIR / "day4_milp_comparison.csv",
        [
            baseline[0],
            planned[0]
        ]
    )

    save_csv(
        OUT_DIR / "day4_milp_by_hour.csv",
        baseline[1] + planned[1]
    )

    save_csv(
        OUT_DIR / "day4_milp_stock_by_station.csv",
        baseline[3] + planned[3]
    )

    save_csv(
        OUT_DIR / "day4_milp_transfers.csv",
        planned[2],
        [
            "hour",
            "origin_id",
            "destination_id",
            "bikes",
            "distance_km",
            "transport_cost"
        ]
    )

    audit = {

        "seed": SEED,
        "scenario": SCENARIO,
        "hours": HOURS,

        "station_count": len(data[0]),

        "neighbor_count": NEIGHBORS,

        "total_move_budget": MOVE_BUDGET,

        "fixed_cost_per_arc_hour":
            FIXED_COST_PER_ARC_HOUR,

        "external_in_cost": EXTRA_COST,
        "external_out_cost": REMOVE_COST,

        "baseline": baseline[0],
        "optimized": planned[0],

        "limitations": [

            "历史借还量为14个训练工作日的小时平均值。",

            "历史已完成订单不等于全部潜在需求。",

            "初始库存为合成情景，"
            "容量来自非同期GBFS元数据。",

            "全部历史借还事件均作为固定流量兑现。",

            "外部补车和疏导为模型假设变量，"
            "不是实际历史运营记录。",

            "小时净额流体近似，"
            "未验证小时内事件的先后顺序。",

            "没有使用订单级OD配对。",

            "站间调拨假设瞬时完成，"
            "没有考虑车辆运输时间和卡车载重。",

            "有向调拨关系不是卡车的完整巡回路线。"
        ]
    }

    (
        OUT_DIR / "day4_milp_audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # ----------------------------
    # 终端结果
    # ----------------------------

    print()
    print(
        "[4/4] 保存结果并完成库存与目标值验收"
    )

    print()
    print("=" * 56)
    print("Day 4 多周期 MILP 结果")
    print("=" * 56)

    for result in [
        baseline[0],
        planned[0]
    ]:

        print(
            f"{result['model']} | "
            f"{result['status']}"
        )

        print(
            f"  目标值="
            f"{result['objective']:.4f}"
        )

        print(
            f"  外部补车="
            f"{result['external_in']:.3f}"
        )

        print(
            f"  外部疏导="
            f"{result['external_out']:.3f}"
        )

        print(
            f"  站间调拨="
            f"{result['moved_bikes']}"
        )

        print(
            f"  启用关系-小时="
            f"{result['active_arc_hours']}"
        )

        print(
            f"  MIP Gap="
            f"{result['mip_gap']:.4%}"
        )

    print()
    print(
        "输出目录：", OUT_DIR
    )


if __name__ == "__main__":
    main()
