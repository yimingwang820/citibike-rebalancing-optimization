
# -*- coding: utf-8 -*-
"""
Day 6-F:
1. 三折留一周末训练稳定性检查
2. 冻结策略在4月25-26日的事件级验证

依赖之前已经运行成功的：
- weekend_preposition.py
- holdout_validation.py
- event_baseline.py
- timeout_simulation.py

直接运行，不需要命令行参数。
"""

from pathlib import Path
from hashlib import sha256
import json

import pandas as pd

import weekend_preposition as wk
import holdout_validation as hold
import event_baseline as event_base
import timeout_simulation as event_model


from project_paths import ROOT

RESULTS = ROOT / "02_RESULTS"

TRAIN_DIR = RESULTS / "DAY6_WEEKEND_TRAIN"
WEEKDAY_DIR = RESULTS / "DAY6_PREPOSITION"

OUT = RESULTS / "DAY6_WEEKEND_VALIDATION"

EVAL_DATES = [
    "2026-04-25",
    "2026-04-26"
]

WEEKEND_PAIRS = [
    ("2026-04-04", "2026-04-05"),
    ("2026-04-11", "2026-04-12"),
    ("2026-04-18", "2026-04-19")
]

TOL = 1e-6


# ==================================================
# 1. 文件指纹
# ==================================================

def fingerprint(path):

    h = sha256()

    with path.open("rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


# ==================================================
# 2. 按指定日期重建训练小时平均流量
# ==================================================

def fold_data(full, selected_dates):

    daily = full["daily"]

    sub = daily.loc[
        daily["date"].isin(selected_dates)
    ]

    expected = (
        len(selected_dates)
        * len(full["ids"])
        * 24
    )

    if len(sub) != expected:
        raise AssertionError(
            "折内每日小时数据不完整"
        )

    means = sub.groupby(
        ["station_id", "hour"],
        as_index=False
    )[
        ["departures", "arrivals"]
    ].mean()

    if len(means) != 25 * 24:
        raise AssertionError(
            "折内平均流量不完整"
        )

    data = dict(full)

    data["D"] = {
        (r.station_id, int(r.hour)):
            float(r.departures)
        for r in means.itertuples(index=False)
    }

    data["A"] = {
        (r.station_id, int(r.hour)):
            float(r.arrivals)
        for r in means.itertuples(index=False)
    }

    return data


# ==================================================
# 3. 读取冻结的周末候选调拨方案
# ==================================================

def load_plan_stock(
    ids,
    capacities,
    initial,
    candidate,
    expected_bikes
):

    invpath = (
        TRAIN_DIR
        / "weekend_candidate_inventories.csv"
    )

    planpath = (
        TRAIN_DIR
        / "weekend_candidate_plans.csv"
    )

    inv = pd.read_csv(
        invpath,
        dtype={"station_id": str}
    )

    plan = pd.read_csv(
        planpath,
        dtype={
            "origin_id": str,
            "destination_id": str
        }
    )

    sub = inv.loc[
        inv["candidate"] == candidate
    ].copy()

    if (
        len(sub) != len(ids)
        or sub["station_id"].duplicated().any()
    ):
        raise AssertionError(
            f"{candidate}库存记录不完整"
        )

    if set(sub["station_id"]) != set(ids):
        raise AssertionError(
            f"{candidate}站点集合不一致"
        )

    stock = {}

    for r in sub.itertuples(index=False):

        old = float(r.before)
        new = float(r.after)

        if abs(
            old - initial[r.station_id]
        ) > TOL:
            raise AssertionError(
                "冻结原始库存不一致"
            )

        if not (
            new.is_integer()
            and 0 <= new <= capacities[r.station_id]
        ):
            raise AssertionError(
                "冻结新库存无效"
            )

        stock[r.station_id] = int(new)

    if sum(stock.values()) != sum(
        initial.values()
    ):
        raise AssertionError(
            "预调拨车辆不守恒"
        )

    horizon = int(
        candidate.split("_")[1].replace("h", "")
    )

    moves = plan.loc[
        plan["horizon_hours"] == horizon
    ].copy()

    if moves[
        ["origin_id", "destination_id"]
    ].duplicated().any():
        raise AssertionError(
            "调拨关系重复"
        )

    if not (
        moves["origin_id"].isin(ids).all()
        and moves["destination_id"].isin(ids).all()
    ):
        raise AssertionError(
            "调拨站点不匹配"
        )

    total = int(moves["bikes"].sum())

    if total != expected_bikes:
        raise AssertionError(
            f"{candidate}调拨数量已改变"
        )

    net = {i: 0 for i in ids}

    for r in moves.itertuples(index=False):

        qty = float(r.bikes)

        if not (
            qty.is_integer()
            and qty > 0
            and r.origin_id != r.destination_id
        ):
            raise AssertionError(
                "存在非法调拨"
            )

        net[r.origin_id] -= int(qty)
        net[r.destination_id] += int(qty)

    for i in ids:

        if stock[i] != initial[i] + net[i]:
            raise AssertionError(
                f"站点{i}库存变动不符"
            )

    cost = (
        float(moves["transport_cost"].sum())
        + wk.FIXED_COST * len(moves)
    )

    return (
        stock,
        total,
        len(moves),
        cost
    )


# ==================================================
# 4. 留一周末交叉验证
# ==================================================

def cross_validate(full):

    print(
        "[2/5] 进行三折留一周末交叉验证..."
    )

    all_dates = set(wk.TRAIN_DATES)

    folds = []
    fold_moves = []

    for k, pair in enumerate(
        WEEKEND_PAIRS,
        start=1
    ):

        held_dates = set(pair)

        train_dates = sorted(
            all_dates - held_dates
        )

        # 4天训练
        training = fold_data(
            full,
            train_dates
        )

        # 2天留出验证
        validation = fold_data(
            full,
            sorted(held_dates)
        )

        # 重新训练24小时MILP
        solution, moves, stock = (
            wk.optimize_weekend(
                training,
                24
            )
        )

        cost = (
            solution["transport_cost"]
            + solution["fixed_cost"]
        )

        # 统一24小时留出数据代理评价
        baseline_score = wk.evaluate_24h(
            validation,
            full["initial"],
            0,
            "baseline"
        )

        optimized_score = wk.evaluate_24h(
            validation,
            stock,
            cost,
            "fold_plan"
        )

        before = float(
            baseline_score[
                "common_24h_surrogate_objective"
            ]
        )

        after = float(
            optimized_score[
                "common_24h_surrogate_objective"
            ]
        )

        folds.append({

            "fold": k,

            "train_dates":
                ",".join(train_dates),

            "heldout_dates":
                ",".join(pair),

            "moves":
                solution["transfer_bikes"],

            "pairs":
                solution["active_pairs"],

            "solver_status":
                solution["status"],

            "mip_gap":
                solution["mip_gap"],

            "validation_baseline_proxy":
                before,

            "validation_strategy_proxy":
                after,

            "validation_proxy_change":
                after - before,

            "validation_proxy_change_pct":
                (
                    (after / before - 1) * 100
                    if before > 0 else None
                )
        })

        fold_moves.extend(
            [
                {"fold": k, **r}
                for r in moves
            ]
        )

        print()
        print(
            f"Fold {k}: "
            f"留出日期={pair}"
        )

        print(
            f"  调拨车辆="
            f"{solution['transfer_bikes']}"
        )

        print(
            f"  调拨关系="
            f"{solution['active_pairs']}"
        )

        print(
            f"  留出代理目标值："
            f"{before:.3f} -> {after:.3f}"
        )

        print(
            f"  变化：{after-before:+.3f}"
        )

    return folds, fold_moves


# ==================================================
# 5. 读取已冻结的工作日方案
# ==================================================

def load_weekday_stock(ids, capacity, initial):

    path = (
        WEEKDAY_DIR
        / "initial_inventory_comparison.csv"
    )

    rows = pd.read_csv(
        path,
        dtype={"station_id": str}
    )

    if (
        len(rows) != 25
        or rows["station_id"].duplicated().any()
        or set(rows["station_id"]) != set(ids)
    ):
        raise AssertionError(
            "工作日库存记录不完整"
        )

    stock = {}

    for r in rows.itertuples(index=False):

        if int(r.before) != initial[r.station_id]:
            raise AssertionError(
                "工作日原始库存不一致"
            )

        if not (
            0 <= int(r.after)
            <= capacity[r.station_id]
        ):
            raise AssertionError(
                "工作日库存越界"
            )

        stock[r.station_id] = int(r.after)

    if sum(stock.values()) != sum(
        initial.values()
    ):
        raise AssertionError(
            "工作日车辆不守恒"
        )

    planfile = (
        WEEKDAY_DIR
        / "training_preposition_plan.csv"
    )

    plan = pd.read_csv(planfile)

    if (
        len(plan) != 6
        or int(plan["bikes"].sum()) != 80
    ):
        raise AssertionError(
            "工作日80辆6关系方案已改变"
        )

    cost = (
        float(plan["transport_cost"].sum())
        + 3.0 * len(plan)
    )

    return stock, cost


# ==================================================
# 6. 4月25–26日事件级回放
# ==================================================

def event_evaluation(ids, cap, configurations):

    print()
    print(
        "[3/5] 读取周末订单并核对小时事件..."
    )

    # 复用Day 6-C已经验收的提取函数，
    # 临时将提取日期缩小至4月25–26日。
    old_dates = hold.DATES
    old_date_set = hold.DATE_SET

    try:

        hold.DATES = list(EVAL_DATES)
        hold.DATE_SET = set(EVAL_DATES)

        trips, extraction = (
            hold.extract_holdout(ids)
        )

        recon, failures = (
            hold.reconcile(trips, ids)
        )

    finally:

        hold.DATES = old_dates
        hold.DATE_SET = old_date_set

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    recon.to_csv(
        OUT / "weekend_hourly_reconciliation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if not failures.empty:

        failures.to_csv(
            OUT / "weekend_hourly_mismatches.csv",
            index=False,
            encoding="utf-8-sig"
        )

        raise AssertionError(
            f"小时事件对账失败：{len(failures)}行"
        )

    print(
        f"PASS：订单对账0差异，"
        f"提取{len(trips):,}条相关订单"
    )

    print()
    print(
        "[4/5] 四种冻结策略进行事件级回放..."
    )

    records = []
    details = []

    original_date = event_base.DATE

    try:

        for date in EVAL_DATES:

            event_base.DATE = date

            starts = (
                trips["start_date"].eq(date)
                & trips["start_station_id"].isin(ids)
            )

            ends = (
                trips["end_date"].eq(date)
                & trips["end_station_id"].isin(ids)
            )

            day_trips = trips.loc[
                starts | ends
            ].copy()

            expected_starts = int(
                starts.sum()
            )

            expected_ends = int(
                ends.sum()
            )

            print()
            print(
                f"测试日期：{date}，"
                f"借车={expected_starts}，"
                f"归还={expected_ends}"
            )

            for label, stock in (
                configurations.items()
            ):

                summary, waits, station_hour = (
                    event_model.replay(
                        ids,
                        cap,
                        stock,
                        day_trips,
                        label
                    )
                )

                # ------------------------------
                # 事件与车辆守恒验收
                # ------------------------------

                if (
                    summary["requests"]
                    != expected_starts
                ):
                    raise AssertionError(
                        f"{date} {label}"
                        "借车请求对账失败"
                    )

                actual_ends = (
                    summary["arrival_attempts"]
                    + summary[
                        "cancelled_internal_arrivals_today"
                    ]
                )

                if actual_ends != expected_ends:
                    raise AssertionError(
                        f"{date} {label}"
                        "归还事件对账失败"
                    )

                if (
                    summary["vehicle_conservation"]
                    != "PASS"
                ):
                    raise AssertionError(
                        "车辆守恒失败"
                    )

                records.append({

                    "date": date,
                    "strategy": label,

                    "borrow_requests":
                        summary["requests"],

                    "borrow_rejected":
                        summary["borrow_rejected"],

                    "arrival_attempts":
                        summary["arrival_attempts"],

                    "return_timeouts":
                        summary["timeout_external_exit"],

                    "target_docked":
                        summary["target_station_docked"],

                    "return_completion_rate":
                        summary[
                            "target_station_docking_rate"
                        ],

                    "wait_occurrences":
                        summary["wait_occurrences"],

                    "observed_wait_minutes":
                        summary["observed_total_wait_min"],

                    "pending_at_midnight":
                        summary["pending_at_midnight"]
                })

                details.extend(
                    [
                        {"date": date, **row}
                        for row in station_hour
                    ]
                )

                print(
                    f"  {label:<20}"
                    f"拒绝={summary['borrow_rejected']:<4}"
                    f"超时={summary['timeout_external_exit']:<4}"
                    f"等待分钟="
                    f"{summary['observed_total_wait_min']:.1f}"
                )

    finally:

        event_base.DATE = original_date

    # --------------------------------------------------
    # 6.1 与已经完成的Day 6-C结果进行回归验收
    # --------------------------------------------------

    frozen_file = (
        RESULTS
        / "DAY6_HOLDOUT"
        / "holdout_daily_comparison.csv"
    )

    if not frozen_file.is_file():
        raise FileNotFoundError(frozen_file)

    old = pd.read_csv(
        frozen_file,
        dtype={"date": str}
    ).set_index("date")

    for r in records:

        if r["strategy"] not in (
            "no_rebalancing",
            "weekday_80"
        ):
            continue

        if r["date"] not in old.index:
            raise AssertionError(
                "缺少已冻结的周末对照记录"
            )

        prefix = (
            "base"
            if r["strategy"] == "no_rebalancing"
            else "plan"
        )

        old_rejected = int(
            old.at[
                r["date"],
                prefix + "_rejected"
            ]
        )

        old_timeout = int(
            old.at[
                r["date"],
                prefix + "_timeouts"
            ]
        )

        if (
            r["borrow_rejected"] != old_rejected
            or r["return_timeouts"] != old_timeout
        ):
            raise AssertionError(
                f"{r['date']} {r['strategy']} "
                "与Day 6-C结果不一致"
            )

    print()
    print(
        "PASS：无调拨和工作日策略"
        "与Day 6-C冻结结果一致"
    )

    return records, details, extraction


# ==================================================
# 7. 主程序
# ==================================================

def main():

    print("=" * 65)
    print(
        "Day 6-F：周末稳定性与事件级策略验证"
    )
    print("=" * 65)

    if event_model.WAIT_LIMIT_MIN != 30:
        raise AssertionError(
            "等待上限已不是30分钟"
        )

    print(
        "[1/5] 检查已冻结的周末训练方案..."
    )

    full = wk.load_training()

    ids = full["ids"]
    cap = full["cap"]
    initial = full["initial"]

    w12, n12, arcs12, cost12 = (
        load_plan_stock(
            ids,
            cap,
            initial,
            "weekend_12h",
            expected_bikes=4
        )
    )

    w24, n24, arcs24, cost24 = (
        load_plan_stock(
            ids,
            cap,
            initial,
            "weekend_24h",
            expected_bikes=9
        )
    )

    if arcs12 != 1 or arcs24 != 2:
        raise AssertionError(
            "周末候选方案关系数已改变"
        )

    weekday, cost_weekday = (
        load_weekday_stock(
            ids,
            cap,
            initial
        )
    )

    print(
        f"W12：{n12}辆/{arcs12}条关系"
    )

    print(
        f"W24：{n24}辆/{arcs24}条关系"
    )

    print(
        "工作日冻结方案：80辆/6条关系"
    )

    # ==============================================
    # 7.1 三折训练稳定性
    # ==============================================

    folds, fold_moves = cross_validate(
        full
    )

    # ==============================================
    # 7.2 四种策略事件仿真
    # ==============================================

    configurations = {

        "no_rebalancing": initial,

        "weekday_80": weekday,

        "weekend_W12": w12,

        "weekend_W24_primary": w24
    }

    daily, station_hours, extraction = (
        event_evaluation(
            ids,
            cap,
            configurations
        )
    )

    # ==============================================
    # 7.3 保存交叉验证结果
    # ==============================================

    print()
    print("[5/5] 保存审计结果...")

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(folds).to_csv(
        OUT / "weekend_3fold_proxy_validation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        fold_moves,
        columns=[
            "fold",
            "horizon_hours",
            "origin_id",
            "destination_id",
            "bikes",
            "distance_km",
            "transport_cost"
        ]
    ).to_csv(
        OUT / "weekend_3fold_plans.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(daily).to_csv(
        OUT / "weekend_strategy_daily.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(station_hours).to_csv(
        OUT / "weekend_strategy_station_hour.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ==============================================
    # 7.4 汇总两天指标
    # ==============================================

    summary_rows = []

    costs = {
        "no_rebalancing": 0.0,
        "weekday_80": cost_weekday,
        "weekend_W12": cost12,
        "weekend_W24_primary": cost24
    }

    for label in configurations:

        subset = [
            r for r in daily
            if r["strategy"] == label
        ]

        arrivals = sum(
            r["arrival_attempts"]
            for r in subset
        )

        docked = sum(
            r["target_docked"]
            for r in subset
        )

        summary_rows.append({

            "strategy": label,

            "days": len(subset),

            "requests": sum(
                r["borrow_requests"]
                for r in subset
            ),

            "borrow_rejected": sum(
                r["borrow_rejected"]
                for r in subset
            ),

            "return_timeouts": sum(
                r["return_timeouts"]
                for r in subset
            ),

            "target_docking_rate": (
                docked / arrivals
                if arrivals else None
            ),

            "observed_wait_minutes": sum(
                r["observed_wait_minutes"]
                for r in subset
            ),

            "nominal_preposition_cost_per_day":
                costs[label]
        })

    pd.DataFrame(summary_rows).to_csv(
        OUT / "weekend_strategy_total.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ==============================================
    # 7.5 审计信息
    # ==============================================

    audit = {

        "training_dates": wk.TRAIN_DATES,

        "crossfold_weekend_pairs":
            WEEKEND_PAIRS,

        "development_dates": EVAL_DATES,

        "primary_weekend_strategy":
            "weekend_W24_primary",

        "secondary_weekend_strategy":
            "weekend_W12",

        "development_data_not_blind": True,

        "initial_bikes_per_day":
            sum(initial.values()),

        "wait_limit_minutes":
            event_model.WAIT_LIMIT_MIN,

        "hourly_reconciliation_mismatches": 0,

        "source_hashes": {

            "weekend_plan":
                fingerprint(
                    TRAIN_DIR
                    / "weekend_candidate_plans.csv"
                ),

            "weekend_inventory":
                fingerprint(
                    TRAIN_DIR
                    / "weekend_candidate_inventories.csv"
                )
        },

        "extraction": extraction,

        "limitations": [

            "交叉验证采用小时平均流量的代理目标，"
            "并非订单级服务效果。",

            "周末训练日期只有6天。",

            "4月25–26日此前已经用于"
            "工作日策略的诊断，不能称为完全盲测。",

            "初始库存为合成库存，每天重新初始化。",

            "站点容量来自非同期GBFS快照。",

            "预调拨未考虑卡车容量和运输时间。",

            "超时外部退出不是实际改道归还。",

            "历史已完成订单不等于真实潜在需求。"
        ]
    }

    (
        OUT / "weekend_validation_audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # ==============================================
    # 7.6 终端最终结果
    # ==============================================

    print()
    print("=" * 65)
    print("Day 6-F：两日事件回放汇总")
    print("=" * 65)

    for r in summary_rows:

        print(
            f"{r['strategy']:<20}"
            f"借车拒绝={r['borrow_rejected']:<5}"
            f"归还超时={r['return_timeouts']:<5}"
            f"归还完成率="
            f"{r['target_docking_rate']:.2%}"
        )

    print()
    print(
        "PASS：训练稳定性与事件级验收完成"
    )

    print(
        "输出目录：", OUT
    )


if __name__ == "__main__":
    main()
