
# -*- coding: utf-8 -*-
"""
Day 5-C：共享单车事件仿真服务质量审计

读取Day 5-B的三个CSV。
计算：
1. 四种库存情景的借车成功率
2. 归还等待发生率
3. 已解决等待的平均时长
4. 缺车与满桩热点站点
5. 小时级服务压力

直接运行，无需额外参数。
"""

from pathlib import Path
import pandas as pd


from citibike.paths import ROOT

INPUT = (
    ROOT / "02_RESULTS"
    / "DAY5_EVENT_BASELINE"
)

OUTPUT = (
    ROOT / "02_RESULTS"
    / "DAY5_SERVICE_AUDIT"
)

SCENARIOS = [
    "low_30",
    "balanced_50",
    "high_70",
    "uneven_50"
]


def require_columns(df, columns, filename):

    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{filename} 缺少列：{sorted(missing)}"
        )


def main():

    print("=" * 60)
    print("Day 5-C：服务质量及站点热点分析")
    print("=" * 60)

    paths = {
        "summary": INPUT / "day5_baseline_summary.csv",
        "hour": INPUT / "day5_baseline_by_hour.csv",
        "station": INPUT / "day5_baseline_by_station.csv"
    }

    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = pd.read_csv(paths["summary"])
    hourly = pd.read_csv(paths["hour"])

    station = pd.read_csv(
        paths["station"],
        dtype={"station_id": str}
    )

    require_columns(
        summary,
        [
            "scenario",
            "initial_bikes",
            "requests_from_area",
            "accepted_departures",
            "rejected_no_bike",
            "arrival_attempts",
            "arrival_full_wait_occurrences",
            "wait_resolved",
            "waiting_at_day_end"
        ],
        "summary"
    )

    require_columns(
        hourly,
        [
            "scenario",
            "hour",
            "requests",
            "accepted",
            "rejected",
            "arrival_attempts",
            "arrive_wait",
            "wait_resolved"
        ],
        "hour"
    )

    require_columns(
        station,
        [
            "scenario",
            "station_id",
            "requests",
            "accepted",
            "rejected",
            "arrival_attempts",
            "arrive_wait",
            "wait_resolved",
            "waiting_minutes_resolved",
            "waiting_at_end"
        ],
        "station"
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 1. 验证各情景的数据一致性
    # ========================================================

    print("[1/4] 检查四个情景的数据一致性...")

    tradeoff = []

    for scenario in SCENARIOS:

        s = summary.loc[
            summary["scenario"] == scenario
        ]

        h = hourly.loc[
            hourly["scenario"] == scenario
        ]

        st = station.loc[
            station["scenario"] == scenario
        ]

        if len(s) != 1:
            raise ValueError(
                f"{scenario} 汇总表记录数异常"
            )

        if len(h) != 24:
            raise ValueError(
                f"{scenario} 小时记录不完整"
            )

        if len(st) != 25:
            raise ValueError(
                f"{scenario} 站点记录不完整"
            )

        if (
            h["hour"].duplicated().any()
            or st["station_id"].duplicated().any()
        ):
            raise ValueError(
                f"{scenario} 存在重复记录"
            )

        r = s.iloc[0]

        checks = {
            "requests": (
                "requests_from_area",
                "requests"
            ),
            "accepted": (
                "accepted_departures",
                "accepted"
            ),
            "rejected": (
                "rejected_no_bike",
                "rejected"
            ),
            "arrival_attempts": (
                "arrival_attempts",
                "arrival_attempts"
            ),
            "wait": (
                "arrival_full_wait_occurrences",
                "arrive_wait"
            ),
            "resolved": (
                "wait_resolved",
                "wait_resolved"
            )
        }

        for name, (summary_col, detail_col) in (
            checks.items()
        ):

            expected = int(r[summary_col])

            if int(h[detail_col].sum()) != expected:
                raise AssertionError(
                    f"{scenario} 小时数据不守恒：{name}"
                )

            if int(st[detail_col].sum()) != expected:
                raise AssertionError(
                    f"{scenario} 站点数据不守恒：{name}"
                )

        if int(st["waiting_at_end"].sum()) != int(
            r["waiting_at_day_end"]
        ):
            raise AssertionError(
                f"{scenario} 日末等待数不一致"
            )

        requests = int(r["requests_from_area"])
        arrivals = int(r["arrival_attempts"])
        waits = int(r["arrival_full_wait_occurrences"])
        resolved = int(r["wait_resolved"])

        total_wait_minutes = float(
            st["waiting_minutes_resolved"].sum()
        )

        tradeoff.append({

            "scenario": scenario,

            "initial_bikes":
                int(r["initial_bikes"]),

            "requests": requests,

            "accepted":
                int(r["accepted_departures"]),

            "rejected":
                int(r["rejected_no_bike"]),

            "borrow_success_rate":
                int(r["accepted_departures"])
                / requests if requests else 0,

            "arrival_attempts":
                arrivals,

            "wait_occurrences":
                waits,

            "arrival_wait_rate":
                waits / arrivals if arrivals else 0,

            "resolved_waits":
                resolved,

            "total_resolved_wait_minutes":
                total_wait_minutes,

            "mean_resolved_wait_minutes":
                total_wait_minutes / resolved
                if resolved else 0,

            "waiting_at_day_end":
                int(r["waiting_at_day_end"])
        })

    result = pd.DataFrame(tradeoff)

    result.to_csv(
        OUTPUT / "day5_service_tradeoff.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("PASS：四个情景的小时与站点统计一致")

    # ========================================================
    # 2. 缺车与满桩热点
    # ========================================================

    print("[2/4] 识别问题最明显的站点...")

    st = station.copy()

    st["borrow_failure_rate"] = (
        st["rejected"]
        / st["requests"].where(
            st["requests"] > 0
        )
    )

    st["arrival_wait_rate"] = (
        st["arrive_wait"]
        / st["arrival_attempts"].where(
            st["arrival_attempts"] > 0
        )
    )

    st["mean_resolved_wait_minutes"] = (
        st["waiting_minutes_resolved"]
        / st["wait_resolved"].where(
            st["wait_resolved"] > 0
        )
    )

    # 无借车或无归还事件的站点保留为NaN，
    # 不把没有观察到的服务率解释成0。
    st.to_csv(
        OUTPUT / "day5_station_service_details.csv",
        index=False,
        encoding="utf-8-sig"
    )

    hotspot_rows = []

    for scenario in SCENARIOS:

        part = st.loc[
            st["scenario"] == scenario
        ]

        shortage_top = part.sort_values(
            ["rejected", "station_id"],
            ascending=[False, True]
        ).head(8)

        waiting_top = part.sort_values(
            ["arrive_wait", "station_id"],
            ascending=[False, True]
        ).head(8)

        for category, top in [
            ("no_bike", shortage_top),
            ("no_dock", waiting_top)
        ]:

            for rank, (_, row) in enumerate(
                top.iterrows(), start=1
            ):

                hotspot_rows.append({
                    "scenario": scenario,
                    "issue": category,
                    "rank": rank,
                    "station_id":
                        row["station_id"],
                    "requests":
                        int(row["requests"]),
                    "rejected":
                        int(row["rejected"]),
                    "arrival_attempts":
                        int(row["arrival_attempts"]),
                    "wait_occurrences":
                        int(row["arrive_wait"]),
                    "mean_resolved_wait_minutes":
                        row["mean_resolved_wait_minutes"],
                    "waiting_at_end":
                        int(row["waiting_at_end"])
                })

    hotspots = pd.DataFrame(hotspot_rows)

    hotspots.to_csv(
        OUTPUT / "day5_station_hotspots.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 3. 小时级压力分析
    # ========================================================

    print("[3/4] 分析借车与归还压力的时段分布...")

    h = hourly.copy()

    h["borrow_failure_rate"] = (
        h["rejected"]
        / h["requests"].where(
            h["requests"] > 0
        )
    )

    h["arrival_wait_rate"] = (
        h["arrive_wait"]
        / h["arrival_attempts"].where(
            h["arrival_attempts"] > 0
        )
    )

    h.to_csv(
        OUTPUT / "day5_hourly_service_details.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 4. 输出报告
    # ========================================================

    print("[4/4] 保存比较报告...")

    lines = [
        "# Day 5-C 服务质量审计",
        "",
        "测试日期：2026-04-21",
        "主区域：5980.10",
        "",
        "## 情景比较",
        "",
        "| 情景 | 借车拒绝 | 借车成功率 | 满桩等待次数 | 归还等待发生率 | 已解决等待平均分钟 | 日末排队 |",
        "|---|---:|---:|---:|---:|---:|---:|"
    ]

    for r in tradeoff:

        lines.append(
            f"| {r['scenario']} "
            f"| {r['rejected']} "
            f"| {r['borrow_success_rate']:.2%} "
            f"| {r['wait_occurrences']} "
            f"| {r['arrival_wait_rate']:.2%} "
            f"| {r['mean_resolved_wait_minutes']:.2f} "
            f"| {r['waiting_at_day_end']} |"
        )

    lines.extend([
        "",
        "## 解释限制",
        "",
        "- 模拟请求源于历史已完成订单，借车拒绝率不等于真实历史缺车率。",
        "- 满桩允许无限时长FIFO等待，因此等待发生不等于最终还车失败。",
        "- 平均等待时间只统计已经解决的等待，不包括日末未解决的等待。",
        "- 缺少每次等待的明细，暂不能计算等待时长中位数或95%分位数。",
        "- 不能仅依据借车成功率确定最佳初始库存。",
        "- 本轮不使用独立测试日调整训练期需求模型参数。",
        ""
    ])

    (
        OUTPUT / "day5_service_audit.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print()
    print("=" * 72)
    print("Day 5-C 服务质量比较")
    print("=" * 72)

    print(
        f"{'Scenario':<14}"
        f"{'Reject':>8}"
        f"{'Waits':>9}"
        f"{'WaitRate':>11}"
        f"{'MeanWaitMin':>14}"
        f"{'EndQueue':>10}"
    )

    for r in tradeoff:

        print(
            f"{r['scenario']:<14}"
            f"{r['rejected']:>8}"
            f"{r['wait_occurrences']:>9}"
            f"{r['arrival_wait_rate']:>10.2%}"
            f"{r['mean_resolved_wait_minutes']:>14.2f}"
            f"{r['waiting_at_day_end']:>10}"
        )

    print()
    print("PASS：服务质量统计与事件账本汇总一致")
    print("输出目录：", OUTPUT)


if __name__ == "__main__":
    main()
