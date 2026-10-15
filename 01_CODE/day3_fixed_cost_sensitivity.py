
# -*- coding: utf-8 -*-
"""
Day 3：固定调拨任务成本敏感性分析

依赖：
    day3_fixed_charge_milp.py
    day3_single_period_lp.py

运行：
    直接运行本文件，不需要命令行参数。

研究：
    F = 0, 1, 3, 5, 10
    不设置路线数量上限。
"""

import csv
import json

import day3_fixed_charge_milp as base


FIXED_COST_VALUES = [0, 1, 3, 5, 10]

# 统一评价成本，用于跨参数方案的公平对照
REFERENCE_FIXED_COST = 3.0

OUTPUT_DIR = (
    base.ROOT
    / "02_RESULTS"
    / "DAY3_FIXED_COST_SENSITIVITY"
)


def save_csv(filepath, rows):
    if not rows:
        return

    with open(
        filepath,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("Day 3：固定成本敏感性分析")
    print("=" * 60)

    # 数据只读取一次，确保各实验使用相同输入
    data = base.load_problem_data()

    results = []
    transfer_plans = []

    for index, F in enumerate(
        FIXED_COST_VALUES, start=1
    ):

        print()
        print("-" * 60)
        print(
            f"[{index}/{len(FIXED_COST_VALUES)}] "
            f"固定任务成本 F = {F}"
        )
        print("-" * 60)

        # 不施加路线数上限
        summary, moves = base.solve_model(
            data=data,
            case_name=f"fixed_cost_{F}",
            fixed_cost=float(F),
            max_routes=None
        )

        # 使用同一固定成本 F=3 对各方案重新评价
        # 注意：该值仅用于统一比较，不是当前模型的目标值
        common_cost_F3 = (
            summary["transport_cost"]
            + REFERENCE_FIXED_COST
            * summary["active_routes"]
            + base.SHORTAGE_PENALTY
            * summary["shortage_proxy"]
        )

        result = {
            "fixed_cost_F": F,
            "status": summary["status"],
            "objective": summary["objective"],
            "shortage_proxy":
                summary["shortage_proxy"],
            "shortage_reduction":
                summary["shortage_reduction"],
            "transfer_bikes":
                summary["transfer_bikes"],
            "active_routes":
                summary["active_routes"],
            "transport_cost":
                summary["transport_cost"],
            "route_fixed_cost":
                summary["route_fixed_cost"],
            "total_relocation_cost":
                summary["total_relocation_cost"],
            "common_evaluation_F3":
                round(common_cost_F3, 6),
            "best_bound":
                summary["best_bound"],
            "mip_gap":
                summary["mip_gap"],
            "runtime_seconds":
                summary["runtime_seconds"],
            "nodes":
                summary["nodes"]
        }

        results.append(result)
        transfer_plans.extend(moves)

        print(
            f"求解状态：{result['status']}"
        )
        print(
            f"目标值：{result['objective']:.5f}"
        )
        print(
            f"缺口代理：{result['shortage_proxy']:.5f}"
        )
        print(
            f"调拨车辆：{result['transfer_bikes']}"
        )
        print(
            f"调拨关系：{result['active_routes']}"
        )
        print(
            f"运输成本：{result['transport_cost']:.5f}"
        )
        print(
            f"固定成本：{result['route_fixed_cost']:.5f}"
        )
        print(
            f"统一 F=3 评价："
            f"{result['common_evaluation_F3']:.5f}"
        )
        print(
            f"MIP Gap：{result['mip_gap']:.4%}"
        )

    # 保存完整结果
    save_csv(
        OUTPUT_DIR / "fixed_cost_sensitivity.csv",
        results
    )

    save_csv(
        OUTPUT_DIR / "fixed_cost_transfer_plans.csv",
        transfer_plans
    )

    with open(
        OUTPUT_DIR / "sensitivity_audit.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "fixed_cost_values": FIXED_COST_VALUES,
                "reference_fixed_cost":
                    REFERENCE_FIXED_COST,
                "max_routes": None,
                "transfer_budget":
                    base.TRANSFER_BUDGET,
                "shortage_penalty":
                    base.SHORTAGE_PENALTY,
                "scenario": base.SCENARIO,
                "target_hour": base.TARGET_HOUR,
                "results": results,
                "notes": [
                    "除固定任务成本外，各模型使用相同输入和约束。",
                    "不设置调拨关系数上限。",
                    "需求为历史已完成订单构造的平均需求代理。",
                    "common_evaluation_F3 为统一成本评价，"
                    "不等于各模型自己的最优目标值。",
                    "求解状态和 MIP Gap 决定结果能否被"
                    "视为全局最优。"
                ]
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 60)
    print("敏感性分析完成")
    print("=" * 60)

    print(
        f"{'F':>4} "
        f"{'Routes':>8} "
        f"{'Bikes':>7} "
        f"{'Shortage':>11} "
        f"{'Objective':>12}"
    )

    for row in results:
        print(
            f"{row['fixed_cost_F']:>4} "
            f"{row['active_routes']:>8} "
            f"{row['transfer_bikes']:>7} "
            f"{row['shortage_proxy']:>11.4f} "
            f"{row['objective']:>12.4f}"
        )

    print()
    print("输出目录：", OUTPUT_DIR)
    print(
        "注意：不同 F 的目标值包含不同固定成本，"
        "不能直接用于判断哪一种 F 更好。"
    )


if __name__ == "__main__":
    main()
