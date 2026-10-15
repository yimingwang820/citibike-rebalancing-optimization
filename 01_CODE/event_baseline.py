
# -*- coding: utf-8 -*-
"""
Day 5-B: 单日真实订单事件驱动仿真（无运营调拨）

处理：
1. 区域内借车，无车则拒绝
2. 内部订单借车失败，取消后续归还
3. 外部流入显式增加本地车辆
4. 区域内向外借车显式减少本地车辆
5. 满桩归还进入FIFO等待队列
6. 跨日已在途订单单独处理
7. 每个事件后检查车辆守恒

直接运行，不需要额外参数。
"""

from collections import defaultdict, deque
import csv
import heapq
import json
from pathlib import Path

import pandas as pd


# ============================================================
# 1. 项目路径
# ============================================================

ROOT = Path(r"D:\BikeSharing_OR_Project")

SEED = "5980.10"
DATE = "2026-04-21"

SCENARIOS = (
    "low_30",
    "balanced_50",
    "high_70",
    "uneven_50"
)

STATIONS_FILE = (
    ROOT / "02_RESULTS"
    / "DAY2_A_FIXED"
    / "day2_selected_candidates.csv"
)

INVENTORY_FILE = (
    ROOT / "02_RESULTS"
    / "DAY2_B_INVENTORY"
    / "initial_inventory_scenarios.csv"
)

TRIPS_FILE = (
    ROOT / "02_RESULTS"
    / "DAY5_EVENT_DATA"
    / "test_day_trip_pairs.csv"
)

OUT_DIR = (
    ROOT / "02_RESULTS"
    / "DAY5_EVENT_BASELINE"
)


# ============================================================
# 2. 读取并验收输入
# ============================================================

def read_inputs():

    for path in (
        STATIONS_FILE,
        INVENTORY_FILE,
        TRIPS_FILE
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到文件：{path}"
            )

    stations = pd.read_csv(
        STATIONS_FILE,
        dtype={
            "station_id": str,
            "candidate_seed": str
        }
    )

    stations = stations.loc[
        stations["candidate_seed"] == SEED
    ].copy()

    if (
        len(stations) != 25
        or stations["station_id"].duplicated().any()
    ):
        raise ValueError(
            "主实验区域必须是25个不同站点"
        )

    ids = sorted(
        stations["station_id"].tolist()
    )

    selected = set(ids)

    capacity = {}

    for r in stations.itertuples(index=False):

        cap = float(r.proxy_capacity)

        if cap <= 0 or not cap.is_integer():
            raise ValueError(
                f"容量无效：{r.station_id}"
            )

        capacity[r.station_id] = int(cap)

    # --------------------------------------------------------
    # 四种库存情景
    # --------------------------------------------------------

    inventory = pd.read_csv(
        INVENTORY_FILE,
        dtype={"station_id": str}
    )

    initial = {}

    for scenario in SCENARIOS:

        part = inventory.loc[
            inventory["scenario"] == scenario
        ]

        if (
            len(part) != 25
            or part["station_id"].duplicated().any()
        ):
            raise ValueError(
                f"库存情景错误：{scenario}"
            )

        if set(part["station_id"]) != selected:
            raise ValueError(
                f"库存站点不一致：{scenario}"
            )

        stocks = {}

        for r in part.itertuples(index=False):

            value = float(r.initial_inventory)

            if (
                not value.is_integer()
                or not 0 <= value <= capacity[r.station_id]
            ):
                raise ValueError(
                    f"初始库存非法：{scenario}"
                )

            stocks[r.station_id] = int(value)

        initial[scenario] = stocks

    # --------------------------------------------------------
    # 订单级数据
    # --------------------------------------------------------

    trips = pd.read_csv(
        TRIPS_FILE,
        dtype={
            "ride_id": str,
            "start_station_id": str,
            "end_station_id": str,
            "start_date": str,
            "end_date": str
        },
        low_memory=False
    )

    required = {
        "ride_id",
        "started_at",
        "ended_at",
        "start_station_id",
        "end_station_id",
        "start_date",
        "end_date"
    }

    missing = required - set(trips.columns)

    if missing:
        raise ValueError(
            f"订单缺少字段：{sorted(missing)}"
        )

    if (
        trips["ride_id"].isna().any()
        or trips["ride_id"].duplicated().any()
    ):
        raise ValueError(
            "订单ID存在缺失或重复"
        )

    trips["started_at"] = pd.to_datetime(
        trips["started_at"],
        errors="raise"
    )

    trips["ended_at"] = pd.to_datetime(
        trips["ended_at"],
        errors="raise"
    )

    if (
        trips["ended_at"] <= trips["started_at"]
    ).any():
        raise ValueError(
            "发现非正行程时长"
        )

    # 时间戳与日期字段一致性
    if (
        trips["start_date"]
        != trips["started_at"].dt.strftime("%Y-%m-%d")
    ).any():
        raise ValueError(
            "开始日期与开始时间不一致"
        )

    if (
        trips["end_date"]
        != trips["ended_at"].dt.strftime("%Y-%m-%d")
    ).any():
        raise ValueError(
            "结束日期与结束时间不一致"
        )

    local_starts = (
        (trips["start_date"] == DATE)
        & trips["start_station_id"].isin(selected)
    )

    local_ends = (
        (trips["end_date"] == DATE)
        & trips["end_station_id"].isin(selected)
    )

    ignored = int(
        (~(local_starts | local_ends)).sum()
    )

    return (
        ids,
        capacity,
        initial,
        trips,
        int(local_starts.sum()),
        int(local_ends.sum()),
        ignored
    )


# ============================================================
# 3. 事件驱动仿真
# ============================================================

def simulate(ids, capacity, initial, trips, scenario):

    selected = set(ids)

    day = pd.Timestamp(DATE)
    tomorrow = day + pd.Timedelta(days=1)

    # 已停靠车辆
    dock = dict(initial)

    # 满桩等待队列
    wait = {
        i: deque()
        for i in ids
    }

    # 本日从区域内出发、且目的站也在区域内的
    # 已成功借出但尚未到达车辆
    underway = set()

    counters = defaultdict(int)

    by_hour = defaultdict(
        lambda: defaultdict(int)
    )

    by_station = {
        i: defaultdict(int)
        for i in ids
    }

    events = []
    ride_lookup = {}

    seq = 0

    # --------------------------------------------------------
    # 3.1 初始化事件队列
    # --------------------------------------------------------

    for r in trips.itertuples(index=False):

        ride_lookup[r.ride_id] = r

        start_today = (
            r.start_date == DATE
            and r.start_station_id in selected
        )

        end_today = (
            r.end_date == DATE
            and r.end_station_id in selected
        )

        if start_today:

            # 内部起点：先安排借车事件。
            # 只有借车成功才安排对应的内部归还。
            heapq.heappush(
                events,
                (
                    r.started_at,
                    0,
                    seq,
                    "START",
                    r.ride_id
                )
            )

            seq += 1

        elif end_today:

            # 本区域外发起，或者前一天已出发：
            # 当日归还视为外生/已有在途车辆到达。
            heapq.heappush(
                events,
                (
                    r.ended_at,
                    1,
                    seq,
                    "EXOGENOUS_ARRIVAL",
                    r.ride_id
                )
            )

            seq += 1

    # --------------------------------------------------------
    # 3.2 每事件车辆守恒
    # --------------------------------------------------------

    def check_conservation():

        for i in ids:

            if not 0 <= dock[i] <= capacity[i]:

                raise AssertionError(
                    f"站点库存越界：{i}"
                )

        waiting = sum(
            len(q)
            for q in wait.values()
        )

        # 区域内全部已知车辆状态
        left = (
            sum(dock.values())
            + len(underway)
            + waiting
        )

        # 外部流入增加本区域车辆，
        # 区域内向外借车减少本区域车辆。
        right = (
            sum(initial.values())
            + counters["exogenous_in"]
            - counters["outbound_departed"]
        )

        if left != right:

            raise AssertionError(
                f"{scenario} 车辆不守恒："
                f"{left} != {right}"
            )

    # --------------------------------------------------------
    # 3.3 满桩等待车辆入桩
    # --------------------------------------------------------

    def fill_waiting(station, now):

        while (
            wait[station]
            and dock[station] < capacity[station]
        ):

            ride_id, arrival_time = (
                wait[station].popleft()
            )

            dock[station] += 1

            counters["wait_resolved"] += 1

            by_hour[now.hour]["wait_resolved"] += 1

            by_station[station]["wait_resolved"] += 1

            wait_minutes = (
                (now - arrival_time)
                .total_seconds() / 60
            )

            by_station[station][
                "waiting_minutes_resolved"
            ] += wait_minutes

    # --------------------------------------------------------
    # 3.4 处理车辆归还
    # --------------------------------------------------------

    def arrive(ride_id, now, is_exogenous):

        r = ride_lookup[ride_id]

        station = r.end_station_id

        if is_exogenous:

            counters["exogenous_in"] += 1

            by_hour[now.hour]["exogenous_in"] += 1

            # 前一天从本区域发起、今天才到达，
            # 属于初始时刻已经在途的车辆。
            if (
                r.start_station_id in selected
                and r.start_date < DATE
            ):

                counters[
                    "prior_day_internal_carry_in"
                ] += 1

        else:

            if ride_id not in underway:

                raise AssertionError(
                    "内部归还没有对应在途车辆"
                )

            underway.remove(ride_id)

        counters["arrival_attempts"] += 1

        by_hour[now.hour][
            "arrival_attempts"
        ] += 1

        by_station[station][
            "arrival_attempts"
        ] += 1

        if (
            dock[station] < capacity[station]
            and not wait[station]
        ):

            # 有空桩，直接入桩。
            dock[station] += 1

            counters["immediate_dock"] += 1

            by_hour[now.hour][
                "immediate_dock"
            ] += 1

            by_station[station][
                "immediate_dock"
            ] += 1

        else:

            # 满桩，进入FIFO等待队列。
            wait[station].append(
                (ride_id, now)
            )

            counters["arrive_wait"] += 1

            by_hour[now.hour][
                "arrive_wait"
            ] += 1

            by_station[station][
                "arrive_wait"
            ] += 1

        fill_waiting(station, now)

    # ========================================================
    # 4. 逐事件执行
    # ========================================================

    check_conservation()

    processed = 0

    while events:

        (
            now,
            priority,
            event_seq,
            kind,
            ride_id
        ) = heapq.heappop(events)

        if not day <= now < tomorrow:

            raise AssertionError(
                f"事件时间越界：{now}"
            )

        r = ride_lookup[ride_id]

        processed += 1

        # ----------------------------------------------------
        # 4.1 借车事件
        # ----------------------------------------------------

        if kind == "START":

            origin = r.start_station_id

            counters["requests"] += 1

            by_hour[now.hour]["requests"] += 1

            by_station[origin]["requests"] += 1

            if dock[origin] == 0:

                # 无车：借车拒绝
                counters["rejected"] += 1

                by_hour[now.hour]["rejected"] += 1

                by_station[origin]["rejected"] += 1

                # 内部订单失败，不得保留对应的归还事件。
                if r.end_station_id in selected:

                    counters[
                        "cancelled_internal_arrivals"
                    ] += 1

                    if r.end_date == DATE:

                        counters[
                            "cancelled_internal_arrivals_today"
                        ] += 1

            else:

                # 借车成功
                dock[origin] -= 1

                counters["accepted"] += 1

                by_hour[now.hour]["accepted"] += 1

                by_station[origin]["accepted"] += 1

                if r.end_station_id in selected:

                    # 内部订单：车辆进入在途状态
                    underway.add(ride_id)

                    if r.ended_at < tomorrow:

                        # 同日归还
                        heapq.heappush(
                            events,
                            (
                                r.ended_at,
                                1,
                                seq,
                                "INTERNAL_ARRIVAL",
                                ride_id
                            )
                        )

                        seq += 1

                    # 跨日归还则保留在途状态，
                    # 不在今天执行明天的归还。

                else:

                    # 车辆驶出25站区域
                    counters[
                        "outbound_departed"
                    ] += 1

                    by_hour[now.hour][
                        "outbound_departed"
                    ] += 1

                # 借走一辆车后可能产生空桩，
                # 让已经等待归还的车辆入桩。
                fill_waiting(origin, now)

        # ----------------------------------------------------
        # 4.2 外生到达
        # ----------------------------------------------------

        elif kind == "EXOGENOUS_ARRIVAL":

            arrive(
                ride_id,
                now,
                is_exogenous=True
            )

        # ----------------------------------------------------
        # 4.3 内部订单归还
        # ----------------------------------------------------

        elif kind == "INTERNAL_ARRIVAL":

            arrive(
                ride_id,
                now,
                is_exogenous=False
            )

        else:

            raise AssertionError(
                f"未知事件类型：{kind}"
            )

        # 每次事件后立即核对车辆守恒
        check_conservation()

    # ========================================================
    # 5. 仿真验收
    # ========================================================

    total_waiting = sum(
        len(q)
        for q in wait.values()
    )

    assert (
        counters["requests"]
        ==
        counters["accepted"] + counters["rejected"]
    )

    assert (
        counters["arrival_attempts"]
        ==
        counters["immediate_dock"]
        + counters["arrive_wait"]
    )

    assert (
        counters["arrive_wait"]
        ==
        counters["wait_resolved"] + total_waiting
    )

    # ========================================================
    # 6. 情景汇总
    # ========================================================

    summary = {

        "scenario": scenario,

        "initial_bikes": sum(
            initial.values()
        ),

        "requests_from_area":
            counters["requests"],

        "accepted_departures":
            counters["accepted"],

        "rejected_no_bike":
            counters["rejected"],

        "acceptance_ratio": round(
            counters["accepted"]
            / max(1, counters["requests"]),
            6
        ),

        "cancelled_internal_arrivals":
            counters["cancelled_internal_arrivals"],

        "cancelled_internal_arrivals_today":
            counters["cancelled_internal_arrivals_today"],

        "exogenous_arrivals":
            counters["exogenous_in"],

        "prior_day_internal_carry_in":
            counters["prior_day_internal_carry_in"],

        "outbound_departed":
            counters["outbound_departed"],

        "arrival_attempts":
            counters["arrival_attempts"],

        "arrival_full_wait_occurrences":
            counters["arrive_wait"],

        "wait_resolved":
            counters["wait_resolved"],

        "waiting_at_day_end":
            total_waiting,

        "internal_in_transit_at_day_end":
            len(underway),

        "docked_at_day_end":
            sum(dock.values()),

        "events_processed":
            processed,

        "conservation_check":
            "PASS"
    }

    # ========================================================
    # 7. 每小时输出
    # ========================================================

    hourly_rows = []

    fields = (
        "requests",
        "accepted",
        "rejected",
        "arrival_attempts",
        "exogenous_in",
        "immediate_dock",
        "arrive_wait",
        "wait_resolved",
        "outbound_departed"
    )

    for hour in range(24):

        rec = by_hour[hour]

        row = {
            "scenario": scenario,
            "hour": hour
        }

        for field in fields:
            row[field] = rec[field]

        hourly_rows.append(row)

    # ========================================================
    # 8. 每站输出
    # ========================================================

    station_rows = []

    station_fields = (
        "requests",
        "accepted",
        "rejected",
        "arrival_attempts",
        "immediate_dock",
        "arrive_wait",
        "wait_resolved",
        "waiting_minutes_resolved"
    )

    for i in ids:

        rec = by_station[i]

        row = {
            "scenario": scenario,
            "station_id": i,
            "capacity": capacity[i],
            "initial_inventory": initial[i],
            "final_docked": dock[i],
            "waiting_at_end": len(wait[i])
        }

        for field in station_fields:
            row[field] = rec[field]

        station_rows.append(row)

    return (
        summary,
        hourly_rows,
        station_rows
    )


# ============================================================
# 9. 保存CSV
# ============================================================

def save_csv(path, rows):

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 10. 主程序
# ============================================================

def main():

    print("=" * 60)
    print("Day 5-B：真实订单事件驱动基线仿真")
    print("=" * 60)

    print(
        "[1/3] 读取订单与四个初始库存情景..."
    )

    (
        ids,
        capacity,
        initial,
        trips,
        expected_requests,
        expected_arrivals,
        ignored
    ) = read_inputs()

    print(
        f"完整订单={len(trips):,}"
    )

    print(
        f"区域当日借车事件={expected_requests:,}"
    )

    print(
        f"区域当日归还事件={expected_arrivals:,}"
    )

    print(
        f"仅保留作证据、本日无区域内事件的订单={ignored}"
    )

    summaries = []
    all_hours = []
    all_stations = []

    for scenario in SCENARIOS:

        print()
        print(
            f"[2/3] 回放库存情景 {scenario}..."
        )

        summary, hours, stations = simulate(
            ids,
            capacity,
            initial[scenario],
            trips,
            scenario
        )

        # 所有本日区域借车请求都应进入回放。
        if (
            summary["requests_from_area"]
            != expected_requests
        ):

            raise AssertionError(
                "借车事件数与原始订单不一致"
            )

        # 原始归还数 =
        # 仿真执行的归还 + 被取消的内部归还。
        if (
            summary["arrival_attempts"]
            + summary["cancelled_internal_arrivals_today"]
            != expected_arrivals
        ):

            raise AssertionError(
                "归还事件取消后对账失败"
            )

        summaries.append(summary)

        all_hours.extend(hours)
        all_stations.extend(stations)

        print(
            f"  借车成功="
            f"{summary['accepted_departures']:,}"
            f"/{expected_requests:,}"
        )

        print(
            f"  无车拒绝="
            f"{summary['rejected_no_bike']:,}"
        )

        print(
            f"  满桩等待发生="
            f"{summary['arrival_full_wait_occurrences']:,}"
        )

        print(
            f"  日末仍在等待入桩="
            f"{summary['waiting_at_day_end']:,}"
        )

        print(
            f"  车辆守恒="
            f"{summary['conservation_check']}"
        )

    # --------------------------------------------------------
    # 保存完整结果
    # --------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_csv(
        OUT_DIR / "day5_baseline_summary.csv",
        summaries
    )

    save_csv(
        OUT_DIR / "day5_baseline_by_hour.csv",
        all_hours
    )

    save_csv(
        OUT_DIR / "day5_baseline_by_station.csv",
        all_stations
    )

    audit = {

        "test_date": DATE,
        "station_seed": SEED,
        "station_count": len(ids),

        "trip_pairs_loaded": len(trips),

        "observed_local_departures":
            expected_requests,

        "observed_local_arrivals":
            expected_arrivals,

        "pairs_without_local_event_today":
            ignored,

        "scenarios": list(SCENARIOS),

        "results": summaries,

        "limitations": [

            "原始订单为历史已完成骑行，"
            "其在仿真中作为反事实借车请求输入。",

            "区域内借车失败时，"
            "相应的区域内归还事件被取消。",

            "区域外发起和前一日已在途的到达"
            "按外生车辆流入处理。",

            "区域内向外借车会使车辆离开本地区域。",

            "满桩车辆允许无限时长FIFO等待，"
            "不是最终的真实运营政策。",

            "时间戳相同时先处理借车，后处理归还。",

            "初始库存是合成库存情景。",

            "当前站点容量来自非同期GBFS快照。",

            "不模拟运营方主动调拨。",

            "没有重建真实历史运营补车记录。",

            "模型内拒绝率不是历史真实缺车率。"
        ]
    }

    (
        OUT_DIR / "day5_baseline_audit.json"
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
        "[3/3] 保存CSV及审计报告..."
    )

    print()
    print("所有库存情景仿真完成。")
    print("输出目录：", OUT_DIR)


if __name__ == "__main__":
    main()
