
# -*- coding: utf-8 -*-
"""
Day 5-D：等待时长与服务热点审计。

复用 Day 5-B 的输入数据和事件处理规则。
记录逐笔等待、分位数、超时风险及站点-小时服务量。
不改变原始仿真结果，不调用 Gurobi。
"""

from collections import defaultdict, deque
import heapq
import json

import numpy as np
import pandas as pd

import event_baseline as base


OUTPUT = (
    base.ROOT / "02_RESULTS"
    / "DAY5_WAIT_DURATION_AUDIT"
)

BASELINE_SUMMARY = (
    base.OUT_DIR / "day5_baseline_summary.csv"
)

WAIT_THRESHOLDS_MIN = (5, 15, 30, 60)


def replay_waits(
    ids,
    capacity,
    initial,
    trips,
    scenario,
    *,
    date=None
):
    if date is None:
        date = base.DATE

    selected = set(ids)

    day = pd.Timestamp(date)
    tomorrow = day + pd.Timedelta(days=1)

    dock = dict(initial)

    waiting = {
        i: deque()
        for i in ids
    }

    underway = set()

    counters = defaultdict(int)

    per_station_hour = defaultdict(
        lambda: defaultdict(int)
    )

    wait_logs = []
    ride_index = {}

    events = []
    seq = 0

    # ===========================================
    # 1. 建立事件队列
    # ===========================================

    def add_event(
        when,
        priority,
        kind,
        ride_id
    ):
        nonlocal seq

        heapq.heappush(
            events,
            (
                when,
                priority,
                seq,
                kind,
                ride_id
            )
        )

        seq += 1

    for r in trips.itertuples(index=False):

        ride_index[r.ride_id] = r

        local_start_today = (
            r.start_date == date
            and r.start_station_id in selected
        )

        local_end_today = (
            r.end_date == date
            and r.end_station_id in selected
        )

        if local_start_today:

            add_event(
                r.started_at,
                0,
                "START",
                r.ride_id
            )

        elif local_end_today:

            add_event(
                r.ended_at,
                1,
                "EXOGENOUS_ARRIVAL",
                r.ride_id
            )

    # ===========================================
    # 2. 车辆守恒
    # ===========================================

    def check_balance():

        dock_count = sum(dock.values())

        queue_count = sum(
            len(q)
            for q in waiting.values()
        )

        assert all(
            0 <= dock[i] <= capacity[i]
            for i in ids
        )

        left = (
            dock_count
            + len(underway)
            + queue_count
        )

        right = (
            sum(initial.values())
            + counters["exogenous_in"]
            - counters["outbound_departed"]
        )

        if left != right:

            raise AssertionError(
                f"{scenario}车辆不守恒："
                f"{left} != {right}"
            )

    # ===========================================
    # 3. FIFO等待队列
    # ===========================================

    def fill_wait(station, now):

        while (
            waiting[station]
            and dock[station] < capacity[station]
        ):

            log_idx = waiting[
                station
            ].popleft()

            row = wait_logs[log_idx]

            dock[station] += 1

            counters["wait_resolved"] += 1

            key = (station, now.hour)

            per_station_hour[
                key
            ]["wait_resolved"] += 1

            row["resolved_at"] = (
                now.isoformat(sep=" ")
            )

            row["status"] = "resolved"

            row["waiting_minutes"] = (
                now - row["_arrived_ts"]
            ).total_seconds() / 60.0

    # ===========================================
    # 4. 执行事件
    # ===========================================

    check_balance()

    while events:

        (
            now,
            _,
            _,
            kind,
            ride_id
        ) = heapq.heappop(events)

        if not day <= now < tomorrow:

            raise AssertionError(
                f"事件时间越界：{now}"
            )

        r = ride_index[ride_id]

        if kind == "START":

            station = r.start_station_id

            counters["requests"] += 1

            per_station_hour[
                (station, now.hour)
            ]["requests"] += 1

            if dock[station] == 0:

                counters["rejected"] += 1

                per_station_hour[
                    (station, now.hour)
                ]["rejected"] += 1

                # 失败的内部借车，取消对应归还
                if r.end_station_id in selected:

                    counters[
                        "cancelled_internal_arrivals"
                    ] += 1

                    if r.end_date == date:

                        counters[
                            "cancelled_internal_arrivals_today"
                        ] += 1

            else:

                dock[station] -= 1

                counters["accepted"] += 1

                per_station_hour[
                    (station, now.hour)
                ]["accepted"] += 1

                if r.end_station_id in selected:

                    underway.add(ride_id)

                    if r.ended_at < tomorrow:

                        add_event(
                            r.ended_at,
                            1,
                            "INTERNAL_ARRIVAL",
                            ride_id
                        )

                else:

                    counters[
                        "outbound_departed"
                    ] += 1

                # 借出车辆释放空桩
                fill_wait(station, now)

        else:

            is_exogenous = (
                kind == "EXOGENOUS_ARRIVAL"
            )

            station = r.end_station_id

            if is_exogenous:

                counters[
                    "exogenous_in"
                ] += 1

            elif kind == "INTERNAL_ARRIVAL":

                if ride_id not in underway:

                    raise AssertionError(
                        "内部归还无对应在途车辆"
                    )

                underway.remove(ride_id)

            else:

                raise AssertionError(
                    f"未知事件类型：{kind}"
                )

            counters[
                "arrival_attempts"
            ] += 1

            per_station_hour[
                (station, now.hour)
            ]["arrival_attempts"] += 1

            if (
                dock[station] < capacity[station]
                and not waiting[station]
            ):

                dock[station] += 1

                counters[
                    "immediate_dock"
                ] += 1

            else:

                row = {
                    "scenario": scenario,
                    "ride_id": ride_id,
                    "station_id": station,

                    "arrived_at":
                        now.isoformat(sep=" "),

                    "resolved_at": "",

                    "status": "waiting",

                    "waiting_minutes": None,

                    "elapsed_until_midnight_minutes":
                        None,

                    "_arrived_ts": now
                }

                waiting[station].append(
                    len(wait_logs)
                )

                wait_logs.append(row)

                counters[
                    "arrive_wait"
                ] += 1

                per_station_hour[
                    (station, now.hour)
                ]["arrive_wait"] += 1

            fill_wait(station, now)

        # 每个事件后验收库存守恒
        check_balance()

    # ===========================================
    # 5. 日末未解决等待：右删失
    # ===========================================

    for row in wait_logs:

        if row["status"] != "resolved":

            row["status"] = (
                "censored_at_day_end"
            )

            row[
                "elapsed_until_midnight_minutes"
            ] = (
                tomorrow - row["_arrived_ts"]
            ).total_seconds() / 60.0

        del row["_arrived_ts"]

    censored = sum(
        row["status"] != "resolved"
        for row in wait_logs
    )

    # ===========================================
    # 6. 账本检查
    # ===========================================

    if counters["arrive_wait"] != (
        counters["wait_resolved"]
        + censored
    ):

        raise AssertionError(
            "等待事件账本不一致"
        )

    if counters["requests"] != (
        counters["accepted"]
        + counters["rejected"]
    ):

        raise AssertionError(
            "借车请求账本不一致"
        )

    if counters["arrival_attempts"] != (
        counters["immediate_dock"]
        + counters["arrive_wait"]
    ):

        raise AssertionError(
            "归还尝试账本不一致"
        )

    # ===========================================
    # 7. 站点-小时数据
    # ===========================================

    station_hour_rows = []

    for station in ids:

        backlog = 0

        for hour in range(24):

            rec = per_station_hour[
                (station, hour)
            ]

            backlog += (
                rec["arrive_wait"]
                - rec["wait_resolved"]
            )

            if backlog < 0:

                raise AssertionError(
                    "小时排队记录不守恒"
                )

            station_hour_rows.append({
                "scenario": scenario,
                "station_id": station,
                "hour": hour,

                "requests":
                    rec["requests"],

                "accepted":
                    rec["accepted"],

                "rejected":
                    rec["rejected"],

                "arrival_attempts":
                    rec["arrival_attempts"],

                "arrive_wait":
                    rec["arrive_wait"],

                "wait_resolved":
                    rec["wait_resolved"],

                "queue_at_hour_end":
                    backlog
            })

        if backlog != len(waiting[station]):

            raise AssertionError(
                "站点日末等待不一致"
            )

    # ===========================================
    # 8. 等待时间分布
    # ===========================================

    resolved_min = np.array(
        [
            row["waiting_minutes"]
            for row in wait_logs
            if row["status"] == "resolved"
        ],
        dtype=float
    )

    threshold_metrics = {}

    for threshold in WAIT_THRESHOLDS_MIN:

        confirmed = int(
            (resolved_min > threshold).sum()
        )

        # 未解决等待若已超过阈值，
        # 也可以确认发生过超时。
        confirmed += sum(
            1
            for row in wait_logs
            if (
                row["status"] != "resolved"
                and row[
                    "elapsed_until_midnight_minutes"
                ] > threshold
            )
        )

        threshold_metrics[
            f"confirmed_wait_over_{threshold}min"
        ] = confirmed

    def percentile(q):

        if len(resolved_min) == 0:
            return None

        return float(
            np.percentile(resolved_min, q)
        )

    peak_area_queue = max(
        sum(
            row["queue_at_hour_end"]
            for row in station_hour_rows
            if row["hour"] == hour
        )
        for hour in range(24)
    )

    summary = {
        "scenario": scenario,

        "requests":
            counters["requests"],

        "accepted":
            counters["accepted"],

        "rejected":
            counters["rejected"],

        "arrival_attempts":
            counters["arrival_attempts"],

        "wait_occurrences":
            counters["arrive_wait"],

        "resolved_waits":
            counters["wait_resolved"],

        "unresolved_waits_at_midnight":
            censored,

        "mean_resolved_min":
            (
                float(resolved_min.mean())
                if len(resolved_min)
                else None
            ),

        "p50_resolved_min":
            percentile(50),

        "p90_resolved_min":
            percentile(90),

        "p95_resolved_min":
            percentile(95),

        "max_resolved_min":
            (
                float(resolved_min.max())
                if len(resolved_min)
                else None
            ),

        "peak_area_hour_end_queue":
            peak_area_queue,

        "docked_end":
            sum(dock.values()),

        "in_transit_end":
            len(underway),

        **threshold_metrics
    }

    return (
        summary,
        wait_logs,
        station_hour_rows
    )


# ==================================================
# 9. 与冻结基线对账
# ==================================================

def validate_against_frozen(
    summary,
    baseline
):

    fields = {
        "requests":
            "requests_from_area",

        "accepted":
            "accepted_departures",

        "rejected":
            "rejected_no_bike",

        "arrival_attempts":
            "arrival_attempts",

        "wait_occurrences":
            "arrival_full_wait_occurrences",

        "resolved_waits":
            "wait_resolved",

        "unresolved_waits_at_midnight":
            "waiting_at_day_end",

        "docked_end":
            "docked_at_day_end",

        "in_transit_end":
            "internal_in_transit_at_day_end"
    }

    for current, previous in fields.items():

        if int(summary[current]) != int(
            baseline[previous]
        ):

            raise AssertionError(
                f"{summary['scenario']} "
                f"{current} 与Day 5-B不一致："
                f"{summary[current]} / "
                f"{baseline[previous]}"
            )

    print(
        f"PASS：{summary['scenario']} "
        "与Day 5-B冻结基线一致"
    )


# ==================================================
# 10. 主程序
# ==================================================

def main():

    print("=" * 60)
    print("Day 5-D：逐笔归还等待时长审计")
    print("=" * 60)

    (
        ids,
        capacity,
        inventories,
        trips,
        _,
        _,
        _
    ) = base.read_inputs()

    if not BASELINE_SUMMARY.is_file():

        raise FileNotFoundError(
            BASELINE_SUMMARY
        )

    frozen = pd.read_csv(
        BASELINE_SUMMARY
    ).set_index("scenario")

    summaries = []
    all_waits = []
    all_station_hours = []

    for scenario in base.SCENARIOS:

        print()
        print("回放情景：", scenario)

        summary, waits, rows = replay_waits(
            ids,
            capacity,
            inventories[scenario],
            trips,
            scenario
        )

        validate_against_frozen(
            summary,
            frozen.loc[scenario]
        )

        summaries.append(summary)
        all_waits.extend(waits)
        all_station_hours.extend(rows)

        print(
            "等待发生次数：",
            summary["wait_occurrences"]
        )

        print(
            "已解决等待平均时长："
            f"{summary['mean_resolved_min']:.2f}分钟"
        )

        print(
            "已解决等待P95："
            f"{summary['p95_resolved_min']:.2f}分钟"
        )

        print(
            "日末未解决等待：",
            summary[
                "unresolved_waits_at_midnight"
            ]
        )

        print(
            "确认超过30分钟的等待次数至少：",
            summary[
                "confirmed_wait_over_30min"
            ]
        )

    # ===========================================
    # 11. 保存输出
    # ===========================================

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(summaries).to_csv(
        OUTPUT / "wait_duration_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(all_waits).to_csv(
        OUTPUT / "wait_event_log.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(all_station_hours).to_csv(
        OUTPUT / "station_hour_service.csv",
        index=False,
        encoding="utf-8-sig"
    )

    audit = {
        "date": base.DATE,
        "scenarios": list(base.SCENARIOS),

        "frozen_baseline_validation":
            "PASS",

        "limitations": [
            "订单来自历史已完成骑行。",
            "等待分位数仅统计已解决等待。",
            "日末未解决记录按右删失处理。",
            "阈值超时统计是确认下界。",
            "尚未改变无限FIFO等待规则。",
            "暂不依据测试日热点训练调拨策略。"
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
    print("Day 5-D审计完成")
    print("输出目录：", OUTPUT)


if __name__ == "__main__":
    main()
