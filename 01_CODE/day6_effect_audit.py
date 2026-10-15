
# -*- coding: utf-8 -*-
"""
Day 6-B：训练期预调拨效果审计

检查：
1. 80辆车辆守恒及6条调拨关系
2. 逐站逐小时结果与总表一致
3. 哪些站点、哪些小时减少了归还超时
4. 哪些站点出现服务恶化
5. 预调拨成本与服务变化

不调用Gurobi，不重新运行事件仿真。
"""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(r"D:\BikeSharing_OR_Project")

INPUT = ROOT / "02_RESULTS" / "DAY6_PREPOSITION"

OUTPUT = (
    ROOT / "02_RESULTS"
    / "DAY6_EFFECT_AUDIT"
)

STATION_FILE = (
    ROOT / "02_RESULTS"
    / "DAY2_A_FIXED"
    / "day2_selected_candidates.csv"
)

BASE = "baseline_balanced_50"
PLAN = "training_preposition"

METRICS = [
    "requests",
    "accepted",
    "rejected",
    "arrival_attempts",
    "wait_occurrences",
    "wait_resolved",
    "timeout_external_exit"
]


def main():

    print("=" * 65)
    print("Day 6-B：预调拨方案与服务变化审计")
    print("=" * 65)

    paths = {
        "plan": INPUT / "training_preposition_plan.csv",
        "inventory": INPUT / "initial_inventory_comparison.csv",
        "summary": INPUT / "devday_service_comparison.csv",
        "detail": INPUT / "devday_station_hour.csv"
    }

    for path in [*paths.values(), STATION_FILE]:
        if not path.is_file():
            raise FileNotFoundError(path)

    moves = pd.read_csv(
        paths["plan"],
        dtype={
            "origin_id": str,
            "destination_id": str
        }
    )

    inventory = pd.read_csv(
        paths["inventory"],
        dtype={"station_id": str}
    )

    summary = pd.read_csv(paths["summary"])

    detail = pd.read_csv(
        paths["detail"],
        dtype={"station_id": str}
    )

    meta = pd.read_csv(
        STATION_FILE,
        dtype={
            "station_id": str,
            "candidate_seed": str
        }
    )

    meta = meta.loc[
        meta["candidate_seed"] == "5980.10",
        ["station_id", "station_name"]
    ]

    # ==================================================
    # 1. 验证调拨方案
    # ==================================================

    print("\n[1/4] 验证调拨数量和初始库存守恒...")

    ids = set(inventory["station_id"])

    if len(ids) != 25:
        raise ValueError("站点数不是25")

    if inventory["station_id"].duplicated().any():
        raise ValueError("初始库存有重复站点")

    if meta["station_id"].duplicated().any():
        raise ValueError("元数据存在重复站点")

    if set(meta["station_id"]) != ids:
        raise ValueError("站点元数据不一致")

    if int(inventory["before"].sum()) != 853:
        raise AssertionError("基线总车辆数不是853")

    if int(inventory["after"].sum()) != 853:
        raise AssertionError("预调拨后车辆不守恒")

    if not (
        (inventory["after"] >= 0).all()
        and (
            inventory["after"]
            <= inventory["capacity"]
        ).all()
    ):
        raise AssertionError("预调拨库存越界")

    if (
        (moves["bikes"] <= 0).any()
        or moves[
            ["origin_id", "destination_id"]
        ].duplicated().any()
    ):
        raise AssertionError(
            "调拨方案存在无效车辆数或重复关系"
        )

    if not (
        moves["origin_id"].isin(ids).all()
        and moves["destination_id"].isin(ids).all()
    ):
        raise AssertionError(
            "调拨出现研究区域之外的站点"
        )

    if (
        moves["origin_id"]
        == moves["destination_id"]
    ).any():
        raise AssertionError("出现站点自身调拨")

    total_moved = int(moves["bikes"].sum())

    if total_moved != 80:
        raise AssertionError(
            f"冻结实验的调拨数量应为80，"
            f"实际={total_moved}"
        )

    if len(moves) != 6:
        raise AssertionError(
            f"冻结实验应有6条调拨关系，"
            f"实际={len(moves)}"
        )

    flow_in = moves.groupby(
        "destination_id"
    )["bikes"].sum()

    flow_out = moves.groupby(
        "origin_id"
    )["bikes"].sum()

    for row in inventory.itertuples(index=False):

        expected_delta = (
            int(flow_in.get(row.station_id, 0))
            - int(flow_out.get(row.station_id, 0))
        )

        if int(row.after - row.before) != (
            expected_delta
        ):
            raise AssertionError(
                f"站点{row.station_id}"
                "调拨前后库存与方案不一致"
            )

    print("PASS：80辆、6条有向关系")
    print("PASS：25站库存及车辆守恒")

    # ==================================================
    # 2. 打印调拨关系
    # ==================================================

    named = moves.merge(
        meta.rename(
            columns={
                "station_id": "origin_id",
                "station_name": "origin_name"
            }
        ),
        on="origin_id",
        validate="many_to_one"
    ).merge(
        meta.rename(
            columns={
                "station_id": "destination_id",
                "station_name": "destination_name"
            }
        ),
        on="destination_id",
        validate="many_to_one"
    )

    named = named.sort_values(
        "bikes",
        ascending=False
    )

    print("\n========== 预调拨方案 ==========")

    print(
        named[
            [
                "origin_id",
                "destination_id",
                "bikes",
                "distance_km",
                "origin_name",
                "destination_name"
            ]
        ].to_string(index=False)
    )

    # 这里只是模型假设成本，非实际货币
    transport_cost = float(
        moves["transport_cost"].sum()
    )

    fixed_cost = 3.0 * len(moves)

    total_cost = transport_cost + fixed_cost

    print(
        f"\n模型运输成本：{transport_cost:.3f}"
    )

    print(
        f"模型固定任务成本：{fixed_cost:.3f}"
    )

    print(
        f"模型调拨总成本：{total_cost:.3f}"
    )

    # ==================================================
    # 3. 逐站逐小时对照
    # ==================================================

    print("\n[2/4] 对齐两套事件回放结果...")

    if set(summary["scenario"]) != {BASE, PLAN}:
        raise ValueError(
            "开发日汇总情景名称不一致"
        )

    if summary["scenario"].duplicated().any():
        raise ValueError("开发日汇总重复")

    baseline = detail.loc[
        detail["scenario"] == BASE
    ].copy()

    optimized = detail.loc[
        detail["scenario"] == PLAN
    ].copy()

    for name, frame in [
        ("baseline", baseline),
        ("optimized", optimized)
    ]:

        if len(frame) != 25 * 24:
            raise ValueError(
                f"{name} 不包含25站×24小时"
            )

        if frame.duplicated(
            ["station_id", "hour"]
        ).any():
            raise ValueError(
                f"{name} 有重复站点小时"
            )

    index_columns = [
        "station_id", "hour"
    ]

    a = baseline.set_index(
        index_columns
    ).sort_index()

    b = optimized.set_index(
        index_columns
    ).sort_index()

    if not a.index.equals(b.index):
        raise AssertionError(
            "基线与优化方案的站点小时不一致"
        )

    if set(a.index.get_level_values(0)) != ids:
        raise AssertionError(
            "事件回放站点与库存站点不一致"
        )

    for metric in METRICS:
        if metric not in a or metric not in b:
            raise ValueError(
                f"缺少服务指标：{metric}"
            )

    # 借车请求必须完全相同
    if not a["requests"].equals(b["requests"]):
        raise AssertionError(
            "不同方案的借车请求不一致"
        )

    renamed_a = a[METRICS].rename(
        columns={
            m: "base_" + m for m in METRICS
        }
    )

    renamed_b = b[METRICS].rename(
        columns={
            m: "plan_" + m for m in METRICS
        }
    )

    joined = renamed_a.join(renamed_b)

    # 检查事件回放明细与汇总一致
    summaries = summary.set_index("scenario")

    checks = [
        ("requests", "requests"),
        ("rejected", "borrow_rejected"),
        (
            "arrival_attempts",
            "arrival_attempts"
        ),
        (
            "wait_occurrences",
            "wait_occurrences"
        ),
        (
            "timeout_external_exit",
            "timeout_external_exit"
        )
    ]

    for scenario, prefix in [
        (BASE, "base_"),
        (PLAN, "plan_")
    ]:

        for metric, summary_metric in checks:

            actual = int(
                joined[prefix + metric].sum()
            )

            expected = int(
                summaries.loc[
                    scenario,
                    summary_metric
                ]
            )

            if actual != expected:
                raise AssertionError(
                    f"{scenario} {metric}对账失败："
                    f"{actual} != {expected}"
                )

    print("PASS：两套服务结果对齐")
    print("PASS：逐站逐小时指标与总表一致")

    # ==================================================
    # 4. 计算站点与小时改善
    # ==================================================

    print("\n[3/4] 计算服务改善与恶化位置...")

    sum_columns = (
        ["base_" + m for m in METRICS]
        + ["plan_" + m for m in METRICS]
    )

    station_effect = (
        joined[sum_columns]
        .groupby(level="station_id")
        .sum()
        .reset_index()
    )

    hour_effect = (
        joined[sum_columns]
        .groupby(level="hour")
        .sum()
        .reset_index()
    )

    for table in [
        station_effect,
        hour_effect
    ]:

        for metric in METRICS:

            table["delta_" + metric] = (
                table["plan_" + metric]
                - table["base_" + metric]
            )

    station_effect = station_effect.merge(
        meta,
        on="station_id",
        validate="one_to_one"
    )

    station_effect = station_effect.merge(
        inventory[
            [
                "station_id",
                "before",
                "after",
                "delta"
            ]
        ],
        on="station_id",
        validate="one_to_one"
    )

    # ----------------------------------------------
    # 4.1 超时改善最多的站点
    # ----------------------------------------------

    improved = station_effect.loc[
        station_effect[
            "delta_timeout_external_exit"
        ] < 0
    ].sort_values(
        "delta_timeout_external_exit"
    )

    worsened = station_effect.loc[
        station_effect[
            "delta_timeout_external_exit"
        ] > 0
    ].sort_values(
        "delta_timeout_external_exit",
        ascending=False
    )

    display_columns = [
        "station_id",
        "station_name",
        "delta",
        "base_timeout_external_exit",
        "plan_timeout_external_exit",
        "delta_timeout_external_exit",
        "delta_rejected"
    ]

    print("\n========== 超时改善最大的站点 ==========")

    if improved.empty:
        print("没有改善的站点")
    else:
        print(
            improved[
                display_columns
            ].head(8).to_string(index=False)
        )

    print("\n========== 超时恶化的站点 ==========")

    if worsened.empty:
        print("没有超时增加的站点")
    else:
        print(
            worsened[
                display_columns
            ].head(8).to_string(index=False)
        )

    # ----------------------------------------------
    # 4.2 小时改善
    # ----------------------------------------------

    print("\n========== 各小时超时变化 ==========")

    print(
        hour_effect[
            [
                "hour",
                "base_timeout_external_exit",
                "plan_timeout_external_exit",
                "delta_timeout_external_exit",
                "delta_rejected"
            ]
        ].to_string(index=False)
    )

    # ==================================================
    # 5. 整体验收
    # ==================================================

    base_row = summaries.loc[BASE]
    plan_row = summaries.loc[PLAN]

    reject_change = int(
        plan_row["borrow_rejected"]
        - base_row["borrow_rejected"]
    )

    timeout_change = int(
        plan_row["timeout_external_exit"]
        - base_row["timeout_external_exit"]
    )

    if (
        int(
            station_effect["delta_rejected"].sum()
        ) != reject_change
        or int(
            station_effect[
                "delta_timeout_external_exit"
            ].sum()
        ) != timeout_change
    ):
        raise AssertionError(
            "站点改善量汇总不一致"
        )

    print("\n========== 全局效果 ==========")

    print(
        f"借车拒绝变化：{reject_change:+d}"
    )

    print(
        f"归还超时变化：{timeout_change:+d}"
    )

    print(
        "目标站完成率变化："
        f"{100 * (plan_row['target_station_docking_rate'] - base_row['target_station_docking_rate']):+.3f}"
        " 个百分点"
    )

    for metric in [
        "wait_occurrences",
        "observed_total_wait_min"
    ]:

        difference = (
            plan_row[metric]
            - base_row[metric]
        )

        print(
            f"{metric} 变化：{difference:+.3f}"
        )

    # ==================================================
    # 6. 保存审计结果
    # ==================================================

    print("\n[4/4] 保存结果...")

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    named.to_csv(
        OUTPUT / "route_details.csv",
        index=False,
        encoding="utf-8-sig"
    )

    station_effect.to_csv(
        OUTPUT / "station_effects.csv",
        index=False,
        encoding="utf-8-sig"
    )

    hour_effect.to_csv(
        OUTPUT / "hour_effects.csv",
        index=False,
        encoding="utf-8-sig"
    )

    audit = {
        "development_date": "2026-04-21",
        "prepositioned_bikes": total_moved,
        "directed_pairs": len(moves),
        "fleet_total": 853,
        "borrow_rejection_change": reject_change,
        "return_timeout_change": timeout_change,
        "synthetic_transfer_cost": round(
            total_cost, 6
        ),
        "stations_with_fewer_timeouts": len(
            improved
        ),
        "stations_with_more_timeouts": len(
            worsened
        ),
        "status": "PASS",
        "notes": [
            "此日期已用于开发验证，不属于最终盲测。",
            "运输和固定任务成本是合成成本单位。",
            "借车请求相同，但内部订单拒绝可能改变归还事件数。",
            "改善量是开发日反事实仿真结果。",
            "本次仅审计结果，不调整优化策略。"
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

    print("\nPASS：Day 6-B审计完成")
    print("输出目录：", OUTPUT)


if __name__ == "__main__":
    main()
