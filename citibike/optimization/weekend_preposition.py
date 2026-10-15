
# -*- coding: utf-8 -*-
"""
Day 6-E：周末专用午夜预调拨 MILP

使用2026年4月的6个训练周末日。
建立12小时和24小时两个候选方案。
统一使用24小时训练代理目标进行评价。

不读取4月21日及4月22-30日测试订单。
不修改原有工作日模型。
"""

from pathlib import Path
from citibike.file_io import file_sha256 as _file_sha256
import json
import math

import gurobipy as gp
from gurobipy import GRB
import pandas as pd


# ==================================================
# 1. 基本设置
# ==================================================

from citibike.paths import ROOT
RESULTS = ROOT / "02_RESULTS"

STATION_FILE = (
    RESULTS / "DAY2_A_FIXED"
    / "day2_selected_candidates.csv"
)

INVENTORY_FILE = (
    RESULTS / "DAY2_B_INVENTORY"
    / "initial_inventory_scenarios.csv"
)

FLOW_FILE = (
    RESULTS / "DAY2_ID_FIXED"
    / "station_flow_summary.csv"
)

OUT = RESULTS / "DAY6_WEEKEND_TRAIN"

SEED = "5980.10"
SCENARIO = "balanced_50"

TRAIN_DATES = [
    "2026-04-04",
    "2026-04-05",
    "2026-04-11",
    "2026-04-12",
    "2026-04-18",
    "2026-04-19"
]

MOVE_BUDGET = 80
NEIGHBORS = 4

FIXED_COST = 3.0
KM_COST = 1.0

SUPPLY_COST = 20.0
EVAC_COST = 20.0

TIME_LIMIT = 120
MIP_GAP = 0.001
TOL = 1e-5


# ==================================================
# 2. 辅助函数
# ==================================================

def sha256(path):
    return _file_sha256(path)


def km(p, q):

    lat1, lon1 = map(math.radians, p)
    lat2, lon2 = map(math.radians, q)

    a = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin((lon2 - lon1) / 2) ** 2
    )

    return (
        2 * 6371.0088
        * math.asin(
            math.sqrt(
                min(1.0, max(0.0, a))
            )
        )
    )


# ==================================================
# 3. 周末训练数据准备
# ==================================================

def load_training():

    for p in (
        STATION_FILE,
        INVENTORY_FILE,
        FLOW_FILE
    ):

        if not p.is_file():
            raise FileNotFoundError(p)

    if any(
        pd.Timestamp(d).weekday() < 5
        for d in TRAIN_DATES
    ):
        raise AssertionError(
            "训练日期混入了工作日"
        )

    st = pd.read_csv(
        STATION_FILE,
        dtype={
            "station_id": str,
            "candidate_seed": str
        }
    )

    st = st.loc[
        st.candidate_seed == SEED
    ].copy()

    inv = pd.read_csv(
        INVENTORY_FILE,
        dtype={"station_id": str}
    )

    inv = inv.loc[
        inv.scenario == SCENARIO
    ].copy()

    ids = sorted(st.station_id.tolist())

    if (
        len(ids) != 25
        or len(set(ids)) != 25
        or len(inv) != 25
    ):
        raise ValueError(
            "主区域或库存不是25站"
        )

    if (
        inv.station_id.duplicated().any()
        or set(inv.station_id) != set(ids)
    ):
        raise ValueError(
            "库存站点集合不一致"
        )

    st = st.set_index("station_id")
    inv = inv.set_index("station_id")

    cap = {}
    initial = {}
    pos = {}

    for i in ids:

        c = float(
            st.at[i, "proxy_capacity"]
        )

        z = float(
            inv.at[i, "initial_inventory"]
        )

        if not (
            math.isfinite(c)
            and math.isfinite(z)
            and c > 0
            and c.is_integer()
            and z.is_integer()
            and 0 <= z <= c
        ):
            raise ValueError(
                f"站点{i}容量或库存非法"
            )

        cap[i] = int(c)
        initial[i] = int(z)

        pos[i] = (
            float(st.at[i, "latitude"]),
            float(st.at[i, "longitude"])
        )

    if (
        sum(initial.values()) != 853
        or sum(cap.values()) != 1706
    ):
        raise AssertionError(
            "冻结库存或容量发生改变"
        )

    # ----------------------------------------------
    # 3.1 读取6个周末日的小时事件
    # ----------------------------------------------

    selected = set(ids)
    dates = set(TRAIN_DATES)

    pieces = []

    reader = pd.read_csv(
        FLOW_FILE,
        dtype={
            "station_id": str,
            "date": str,
            "type": str,
            "hour": "int16",
            "count": "int64"
        },
        chunksize=250000
    )

    for chunk in reader:

        sub = chunk.loc[
            chunk.date.isin(dates)
            & chunk.station_id.isin(selected),
            [
                "date",
                "station_id",
                "hour",
                "type",
                "count"
            ]
        ]

        if sub.empty:
            continue

        if (
            (sub["count"] < 0).any()
            or not sub.hour.between(0, 23).all()
            or not sub.type.isin(
                ["start", "end"]
            ).all()
        ):
            raise ValueError(
                "训练小时流量存在非法数据"
            )

        pieces.append(sub)

    if not pieces:
        raise RuntimeError(
            "未读取到周末训练流量"
        )

    raw = pd.concat(
        pieces,
        ignore_index=True
    )

    agg = raw.groupby(
        [
            "date",
            "station_id",
            "hour",
            "type"
        ],
        as_index=False
    )["count"].sum()

    wide = agg.pivot_table(
        index=[
            "date",
            "station_id",
            "hour"
        ],
        columns="type",
        values="count",
        aggfunc="sum",
        fill_value=0
    )

    for event_type in [
        "start",
        "end"
    ]:

        if event_type not in wide.columns:
            wide[event_type] = 0

    # 补齐零流量时段
    index = pd.MultiIndex.from_product(
        [
            TRAIN_DATES,
            ids,
            range(24)
        ],
        names=[
            "date",
            "station_id",
            "hour"
        ]
    )

    daily = (
        wide[["start", "end"]]
        .reindex(index, fill_value=0)
        .reset_index()
    )

    daily.columns.name = None

    daily = daily.rename(
        columns={
            "start": "departures",
            "end": "arrivals"
        }
    )

    if (
        len(daily) != 6 * 25 * 24
        or int(
            daily[
                ["departures", "arrivals"]
            ].to_numpy().sum()
        ) != int(raw["count"].sum())
    ):
        raise AssertionError(
            "周末事件计数不守恒"
        )

    # ----------------------------------------------
    # 3.2 生成周末平均借还流量
    # ----------------------------------------------

    profile = daily.groupby(
        ["station_id", "hour"],
        as_index=False
    )[
        ["departures", "arrivals"]
    ].mean()

    profile = profile.rename(
        columns={
            "departures": "avg_departures",
            "arrivals": "avg_arrivals"
        }
    )

    if len(profile) != 25 * 24:
        raise AssertionError(
            "周末平均数据不完整"
        )

    D = {
        (r.station_id, int(r.hour)):
            float(r.avg_departures)
        for r in profile.itertuples(index=False)
    }

    A = {
        (r.station_id, int(r.hour)):
            float(r.avg_arrivals)
        for r in profile.itertuples(index=False)
    }

    # ----------------------------------------------
    # 3.3 生成候选调拨网络
    # ----------------------------------------------

    arcs = []
    distance = {}

    for i in ids:

        nearby = sorted(
            (
                j for j in ids
                if j != i
            ),
            key=lambda j: (
                km(pos[i], pos[j]),
                j
            )
        )[:NEIGHBORS]

        for j in nearby:

            M = min(
                initial[i],
                cap[j] - initial[j],
                MOVE_BUDGET
            )

            if M < 1:
                continue

            arcs.append((i, j))

            distance[i, j] = km(
                pos[i], pos[j]
            )

    return {
        "ids": ids,
        "cap": cap,
        "initial": initial,
        "D": D,
        "A": A,
        "arcs": arcs,
        "distance": distance,
        "daily": daily,
        "profile": profile,
        "events": int(raw["count"].sum())
    }


# ==================================================
# 4. 周末预调拨 MILP
# ==================================================

def optimize_weekend(data, horizon):

    ids = data["ids"]
    cap = data["cap"]
    initial = data["initial"]

    D = data["D"]
    A = data["A"]

    arcs = data["arcs"]
    distance = data["distance"]

    T = horizon

    model = gp.Model(
        f"weekend_preposition_{T}h"
    )

    model.Params.OutputFlag = 0
    model.Params.TimeLimit = TIME_LIMIT
    model.Params.MIPGap = MIP_GAP

    # ----------------------------------------------
    # 4.1 变量
    # ----------------------------------------------

    stock = model.addVars(
        ids,
        range(T + 1),
        lb=0,
        name="inventory"
    )

    supply = model.addVars(
        ids,
        range(T),
        lb=0,
        name="virtual_supply"
    )

    x = model.addVars(
        arcs,
        vtype=GRB.INTEGER,
        lb=0,
        name="transfer"
    )

    y = model.addVars(
        arcs,
        vtype=GRB.BINARY,
        name="activate"
    )

    incoming = {
        i: [
            j for j in ids
            if (j, i) in distance
        ]
        for i in ids
    }

    outgoing = {
        i: [
            j for j in ids
            if (i, j) in distance
        ]
        for i in ids
    }

    # ----------------------------------------------
    # 4.2 固定费用与调拨预算
    # ----------------------------------------------

    for i, j in arcs:

        M = min(
            initial[i],
            cap[j] - initial[j],
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

    # ----------------------------------------------
    # 4.3 午夜调拨后的初始库存
    # ----------------------------------------------

    for i in ids:

        sent = gp.quicksum(
            x[i, j]
            for j in outgoing[i]
        )

        received = gp.quicksum(
            x[j, i]
            for j in incoming[i]
        )

        model.addConstr(
            sent <= initial[i]
        )

        model.addConstr(
            received <= cap[i] - initial[i]
        )

        model.addConstr(
            stock[i, 0]
            == initial[i] + received - sent
        )

        # 全时段站点容量约束
        for t in range(T + 1):

            stock[i, t].UB = cap[i]

        # ------------------------------------------
        # 4.4 分阶段库存约束
        # ------------------------------------------

        for t in range(T):

            # 虚拟补车不提前储备超出本小时需求的车
            supply[i, t].UB = D[i, t]

            # 借车阶段不得提前使用归还车辆
            model.addConstr(
                stock[i, t] + supply[i, t]
                >= D[i, t]
            )

            # 归还后库存；差额为非负的虚拟疏导量
            model.addConstr(
                stock[i, t + 1]
                <= stock[i, t]
                + supply[i, t]
                - D[i, t]
                + A[i, t]
            )

    # ----------------------------------------------
    # 4.5 目标函数
    # ----------------------------------------------

    transport_expr = gp.quicksum(
        KM_COST
        * distance[i, j]
        * x[i, j]
        for i, j in arcs
    )

    fixed_expr = FIXED_COST * gp.quicksum(
        y[i, j]
        for i, j in arcs
    )

    supply_expr = gp.quicksum(
        supply[i, t]
        for i in ids
        for t in range(T)
    )

    # 隐式外部疏导量
    evacuation_expr = gp.quicksum(
        stock[i, t]
        + supply[i, t]
        - D[i, t]
        + A[i, t]
        - stock[i, t + 1]
        for i in ids
        for t in range(T)
    )

    model.setObjective(
        transport_expr
        + fixed_expr
        + SUPPLY_COST * supply_expr
        + EVAC_COST * evacuation_expr,
        GRB.MINIMIZE
    )

    model.update()

    # 受限许可证问题规模检查
    if (
        model.NumVars > 2000
        or model.NumConstrs > 2000
    ):
        raise RuntimeError(
            "模型规模过大："
            f"{model.NumVars}变量，"
            f"{model.NumConstrs}约束"
        )

    # ----------------------------------------------
    # 4.6 Gurobi求解
    # ----------------------------------------------

    model.optimize()

    if model.SolCount == 0:

        raise RuntimeError(
            f"{T}小时模型没有可行解，"
            f"Status={model.Status}"
        )

    if model.Status not in (
        GRB.OPTIMAL,
        GRB.TIME_LIMIT
    ):
        raise RuntimeError(
            f"{T}小时模型异常中止："
            f"Status={model.Status}"
        )

    # ----------------------------------------------
    # 4.7 提取调拨结果
    # ----------------------------------------------

    moves = []

    after = dict(initial)

    for i, j in arcs:

        qty = x[i, j].X

        if qty <= TOL:
            continue

        if (
            abs(qty - round(qty)) > TOL
            or y[i, j].X < 1 - TOL
        ):
            raise AssertionError(
                "调拨整数或关系变量异常"
            )

        qty = int(round(qty))

        moves.append({
            "horizon_hours": T,
            "origin_id": i,
            "destination_id": j,
            "bikes": qty,
            "distance_km": round(
                distance[i, j], 7
            ),
            "transport_cost": round(
                KM_COST * distance[i, j] * qty,
                7
            )
        })

        after[i] -= qty
        after[j] += qty

    if sum(after.values()) != sum(
        initial.values()
    ):
        raise AssertionError(
            "预调拨总库存不守恒"
        )

    for i in ids:

        if not 0 <= after[i] <= cap[i]:
            raise AssertionError(
                f"站点{i}库存越界"
            )

        if abs(
            stock[i, 0].X - after[i]
        ) > TOL:
            raise AssertionError(
                "调拨库存与模型解不一致"
            )

    total_moves = sum(
        r["bikes"] for r in moves
    )

    if total_moves > MOVE_BUDGET:
        raise AssertionError(
            "超过80辆调拨预算"
        )

    # ----------------------------------------------
    # 4.8 目标函数分解
    # ----------------------------------------------

    S = sum(
        supply[i, t].X
        for i in ids
        for t in range(T)
    )

    E = sum(
        stock[i, t].X
        + supply[i, t].X
        - D[i, t]
        + A[i, t]
        - stock[i, t + 1].X
        for i in ids
        for t in range(T)
    )

    if E < -TOL:
        raise AssertionError(
            "虚拟疏导出现负值"
        )

    move_cost = sum(
        KM_COST
        * distance[
            r["origin_id"],
            r["destination_id"]
        ]
        * r["bikes"]
        for r in moves
    )

    task_cost = FIXED_COST * len(moves)

    reconstructed = (
        move_cost
        + task_cost
        + SUPPLY_COST * S
        + EVAC_COST * E
    )

    if abs(
        model.ObjVal - reconstructed
    ) > 1e-3:
        raise AssertionError(
            "模型目标函数分解不一致"
        )

    summary = {
        "horizon_hours": T,
        "status": (
            "OPTIMAL_WITH_TOLERANCE"
            if model.Status == GRB.OPTIMAL
            else "TIME_LIMIT_WITH_INCUMBENT"
        ),
        "surrogate_objective":
            model.ObjVal,
        "virtual_supply": S,
        "virtual_evacuation": E,
        "transfer_bikes": total_moves,
        "active_pairs": len(moves),
        "transport_cost": move_cost,
        "fixed_cost": task_cost,
        "mip_gap": model.MIPGap,
        "runtime_seconds": model.Runtime,
        "num_variables": model.NumVars,
        "num_constraints": model.NumConstrs
    }

    return summary, moves, after


# ==================================================
# 5. 统一24小时训练期评价
# ==================================================

def evaluate_24h(
    data,
    inventory,
    transfer_cost,
    label
):

    ids = data["ids"]
    cap = data["cap"]

    D = data["D"]
    A = data["A"]

    T = 24

    model = gp.Model(
        "common_training_evaluation"
    )

    model.Params.OutputFlag = 0

    I = model.addVars(
        ids,
        range(T + 1),
        lb=0,
        name="inventory"
    )

    S = model.addVars(
        ids,
        range(T),
        lb=0,
        name="supply"
    )

    for i in ids:

        model.addConstr(
            I[i, 0] == inventory[i]
        )

        for t in range(T + 1):

            I[i, t].UB = cap[i]

        for t in range(T):

            S[i, t].UB = D[i, t]

            model.addConstr(
                I[i, t] + S[i, t]
                >= D[i, t]
            )

            model.addConstr(
                I[i, t + 1]
                <= I[i, t]
                + S[i, t]
                - D[i, t]
                + A[i, t]
            )

    evacuation = gp.quicksum(
        I[i, t]
        + S[i, t]
        - D[i, t]
        + A[i, t]
        - I[i, t + 1]
        for i in ids
        for t in range(T)
    )

    model.setObjective(
        SUPPLY_COST * gp.quicksum(
            S[i, t]
            for i in ids
            for t in range(T)
        )
        + EVAC_COST * evacuation,
        GRB.MINIMIZE
    )

    model.optimize()

    if model.Status != GRB.OPTIMAL:

        raise RuntimeError(
            f"24小时统一评价失败：{label}"
        )

    supply = sum(
        S[i, t].X
        for i in ids
        for t in range(T)
    )

    evac = sum(
        I[i, t].X
        + S[i, t].X
        - D[i, t]
        + A[i, t]
        - I[i, t + 1].X
        for i in ids
        for t in range(T)
    )

    return {
        "candidate": label,
        "training_24h_virtual_supply": supply,
        "training_24h_virtual_evacuation": evac,
        "preposition_cost_units": transfer_cost,
        "common_24h_surrogate_objective":
            model.ObjVal + transfer_cost
    }


# ==================================================
# 6. 主程序
# ==================================================

def main():

    print("=" * 64)
    print("Day 6-E：周末专用预调拨训练")
    print("=" * 64)

    # ----------------------------------------------
    # 6.1 加载周末训练数据
    # ----------------------------------------------

    print(
        "[1/4] 读取6个训练周末日的历史流量..."
    )

    data = load_training()

    print(
        f"研究站点=25，"
        f"训练周末日=6，"
        f"初始车辆=853"
    )

    print(
        "训练站点借还事件合计："
        f"{data['events']:,}"
    )

    area = data["profile"].groupby(
        "hour",
        as_index=False
    )[
        ["avg_departures", "avg_arrivals"]
    ].sum()

    print()
    print("周末区域平均小时流量：")

    print(
        area.to_string(
            index=False,
            float_format=lambda v: f"{v:.2f}"
        )
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    data["daily"].to_csv(
        OUT / "weekend_train_station_day_hour.csv",
        index=False,
        encoding="utf-8-sig"
    )

    data["profile"].to_csv(
        OUT / "weekend_train_station_hour_profile.csv",
        index=False,
        encoding="utf-8-sig"
    )

    area.to_csv(
        OUT / "weekend_train_area_hour_profile.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------------------------------
    # 6.2 求解两种训练窗口
    # ----------------------------------------------

    print()
    print(
        "[2/4] 求解12小时和24小时周末MILP..."
    )

    outcomes = []
    all_moves = []
    all_inventories = []

    solutions = {}

    for h in [12, 24]:

        summary, moves, after = (
            optimize_weekend(data, h)
        )

        outcomes.append(summary)
        all_moves.extend(moves)

        solutions[h] = (
            summary,
            after
        )

        for i in data["ids"]:

            all_inventories.append({
                "candidate": f"weekend_{h}h",
                "station_id": i,
                "capacity": data["cap"][i],
                "before": data["initial"][i],
                "after": after[i],
                "delta": (
                    after[i] - data["initial"][i]
                )
            })

        print(
            f"\n方案 W{h}:"
        )

        print(
            f"  求解状态：{summary['status']}"
        )

        print(
            "  训练代理目标值："
            f"{summary['surrogate_objective']:.3f}"
        )

        print(
            "  预调拨车辆："
            f"{summary['transfer_bikes']}"
        )

        print(
            "  有向调拨关系："
            f"{summary['active_pairs']}"
        )

        print(
            f"  MIP Gap："
            f"{summary['mip_gap']:.4%}"
        )

        print(
            "  模型规模："
            f"{summary['num_variables']}变量 / "
            f"{summary['num_constraints']}约束"
        )

    # ----------------------------------------------
    # 6.3 统一24小时训练评价
    # ----------------------------------------------

    print()
    print(
        "[3/4] 统一24小时训练代理评价..."
    )

    common = [
        evaluate_24h(
            data,
            data["initial"],
            0.0,
            "no_preposition"
        )
    ]

    for h in [12, 24]:

        summary, after = solutions[h]

        cost = (
            summary["transport_cost"]
            + summary["fixed_cost"]
        )

        common.append(
            evaluate_24h(
                data,
                after,
                cost,
                f"weekend_{h}h"
            )
        )

    for row in common:

        print(
            f"{row['candidate']:<18} "
            "统一24小时目标值="
            f"{row['common_24h_surrogate_objective']:.3f}"
        )

        print(
            "  虚拟补车="
            f"{row['training_24h_virtual_supply']:.3f}，"
            "虚拟疏导="
            f"{row['training_24h_virtual_evacuation']:.3f}"
        )

    # ----------------------------------------------
    # 6.4 保存结果
    # ----------------------------------------------

    print()
    print(
        "[4/4] 保存周末模型与调拨方案..."
    )

    pd.DataFrame(outcomes).to_csv(
        OUT / "weekend_training_model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        all_moves,
        columns=[
            "horizon_hours",
            "origin_id",
            "destination_id",
            "bikes",
            "distance_km",
            "transport_cost"
        ]
    ).to_csv(
        OUT / "weekend_candidate_plans.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(all_inventories).to_csv(
        OUT / "weekend_candidate_inventories.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(common).to_csv(
        OUT / "weekend_common_24h_training_score.csv",
        index=False,
        encoding="utf-8-sig"
    )

    audit = {

        "training_weekend_dates": TRAIN_DATES,

        "used_heldout_dates": False,

        "primary_candidate_for_next_stage":
            "weekend_24h",

        "secondary_candidate":
            "weekend_12h",

        "selected_station_seed": SEED,

        "scenario": SCENARIO,

        "initial_bikes": sum(
            data["initial"].values()
        ),

        "total_capacity_proxy": sum(
            data["cap"].values()
        ),

        "max_preposition_bikes": MOVE_BUDGET,

        "max_nearest_outgoing_arcs": NEIGHBORS,

        "max_wait_minutes_in_future_replay": 30,

        "source_files_sha256": {
            "station_file":
                sha256(STATION_FILE),

            "inventory_file":
                sha256(INVENTORY_FILE)
        },

        "license_size_safe": all(
            r["num_variables"] <= 2000
            and r["num_constraints"] <= 2000
            for r in outcomes
        ),

        "model_explanation": (
            "历史完成骑行平均流量，"
            "合成初始库存，"
            "虚拟补车和疏导代理成本。"
        ),

        "limitations": [
            "仅6个周末训练日，样本较少。",
            "历史完成订单不代表全部潜在需求。",
            "没有真实初始库存及运输成本。",
            "预调拨忽略卡车容量和运输时间。",
            "12小时与24小时原始目标值不可直接比较。",
            "不使用4月25–26日选择策略。",
            "后续需要训练期留一周末稳定性检查。"
        ]
    }

    (
        OUT / "weekend_training_audit.json"
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
        "PASS：周末训练模型及候选方案生成完成"
    )

    print(
        "注意：尚未进行订单级独立验证。"
    )

    print("输出目录：", OUT)


if __name__ == "__main__":
    main()
