
# -*- coding: utf-8 -*-
"""
Day 5-E：最大等待30分钟的事件驱动仿真。

规则：
1. 借车无车则拒绝
2. 内部订单借车失败则取消对应归还
3. 满桩归还进入FIFO队列
4. 借车释放空桩，优先服务最早等待的车辆
5. 等待达到30分钟仍无法入桩，退出研究区域
6. 午夜前尚未超时，保留为右删失记录
7. 每次事件后进行车辆守恒检查

直接运行，无需命令行参数。
"""

from collections import defaultdict, deque
import heapq
import json

import numpy as np
import pandas as pd

import day5_event_baseline as base


# ==================================================
# 1. 参数和路径
# ==================================================

WAIT_LIMIT_MIN = 30

OUT_DIR = (
    base.ROOT
    / "02_RESULTS"
    / "DAY5_TIMEOUT_30MIN"
)

TOL = 1e-9


# ==================================================
# 2. 单情景事件仿真
# ==================================================

def replay(ids, capacity, initial, trips, scenario):

    selected = set(ids)

    day = pd.Timestamp(base.DATE)
    end_day = day + pd.Timedelta(days=1)

    patience = pd.Timedelta(
        minutes=WAIT_LIMIT_MIN
    )

    dock = dict(initial)

    queue = {
        i: deque()
        for i in ids
    }

    # 当天成功借出的内部在途车辆
    in_transit = set()

    counts = defaultdict(int)

    station_hour = defaultdict(
        lambda: defaultdict(int)
    )

    ride_map = {}

    wait_logs = []
    events = []

    serial = 0

    # ------------------------------------------------
    # 2.1 事件安排
    # ------------------------------------------------

    def schedule(ts, priority, kind, key):

        nonlocal serial

        if day <= ts < end_day:

            heapq.heappush(
                events,
                (
                    ts,
                    priority,
                    serial,
                    kind,
                    key
                )
            )

            serial += 1

    for r in trips.itertuples(index=False):

        ride_map[r.ride_id] = r

        start_today = (
            r.start_date == base.DATE
            and r.start_station_id in selected
        )

        end_today = (
            r.end_date == base.DATE
            and r.end_station_id in selected
        )

        if start_today:

            schedule(
                r.started_at,
                0,
                "START",
                r.ride_id
            )

        elif end_today:

            schedule(
                r.ended_at,
                1,
                "EXTERNAL_ARRIVAL",
                r.ride_id
            )

    # ------------------------------------------------
    # 2.2 车辆守恒
    # ------------------------------------------------

    def conservation():

        for i in ids:

            if not 0 <= dock[i] <= capacity[i]:

                raise AssertionError(
                    f"站点库存越界：{i}"
                )

        on_hand = (
            sum(dock.values())
            + len(in_transit)
            + sum(
                len(q)
                for q in queue.values()
            )
        )

        # 外部进入显式增加车辆
        # 区域外流出及超时退出显式减少车辆
        expected = (
            sum(initial.values())
            + counts["exogenous_in"]
            - counts["outbound_departed"]
            - counts["timeout_external_exit"]
        )

        if on_hand != expected:

            raise AssertionError(
                f"{scenario}车辆守恒失败："
                f"{on_hand} != {expected}"
            )

    # ------------------------------------------------
    # 2.3 FIFO等待队列
    # ------------------------------------------------

    def fill_waiting(station, now):

        while (
            queue[station]
            and dock[station] < capacity[station]
        ):

            idx = queue[station].popleft()

            row = wait_logs[idx]

            if row["status"] != "waiting":

                raise AssertionError(
                    "等待队列中存在已退出车辆"
                )

            dock[station] += 1

            row["status"] = "docked_after_wait"

            row["exit_at"] = str(now)

            row["observed_wait_minutes"] = (
                now - row["_arrived"]
            ).total_seconds() / 60.0

            counts["wait_resolved"] += 1

            station_hour[
                (station, now.hour)
            ]["wait_resolved"] += 1

    # ==================================================
    # 3. 事件回放
    # ==================================================

    conservation()

    while events:

        (
            now,
            priority,
            event_seq,
            kind,
            key
        ) = heapq.heappop(events)

        counts["events_processed"] += 1

        # --------------------------------------------
        # 3.1 等待超时
        # --------------------------------------------

        if kind == "TIMEOUT":

            idx = key

            row = wait_logs[idx]

            # 如果车辆已经成功入桩，则忽略超时事件
            if row["status"] != "waiting":

                conservation()
                continue

            station = row["station_id"]

            # 相同等待上限下，最早入队者最早超时
            if (
                not queue[station]
                or queue[station][0] != idx
            ):

                raise AssertionError(
                    "FIFO超时顺序异常"
                )

            queue[station].popleft()

            row["status"] = "timeout_external_exit"

            row["exit_at"] = str(now)

            row["observed_wait_minutes"] = (
                now - row["_arrived"]
            ).total_seconds() / 60.0

            # 明确计入区域外退出账本
            counts["timeout_external_exit"] += 1

            station_hour[
                (station, now.hour)
            ]["timeout_external_exit"] += 1

        # --------------------------------------------
        # 3.2 借车事件
        # --------------------------------------------

        elif kind == "START":

            r = ride_map[key]

            origin = r.start_station_id

            counts["requests"] += 1

            station_hour[
                (origin, now.hour)
            ]["requests"] += 1

            if dock[origin] == 0:

                counts["rejected"] += 1

                station_hour[
                    (origin, now.hour)
                ]["rejected"] += 1

                # 内部借车失败：
                # 不生成其对应的归还事件
                if (
                    r.end_station_id in selected
                    and r.end_date == base.DATE
                ):

                    counts[
                        "cancelled_internal_arrivals_today"
                    ] += 1

            else:

                dock[origin] -= 1

                counts["accepted"] += 1

                station_hour[
                    (origin, now.hour)
                ]["accepted"] += 1

                if r.end_station_id in selected:

                    in_transit.add(key)

                    # 跨日到达不在当天执行
                    schedule(
                        r.ended_at,
                        1,
                        "INTERNAL_ARRIVAL",
                        key
                    )

                else:

                    counts[
                        "outbound_departed"
                    ] += 1

                # 借车释放了一个空桩
                fill_waiting(origin, now)

        # --------------------------------------------
        # 3.3 归还事件
        # --------------------------------------------

        elif kind in (
            "EXTERNAL_ARRIVAL",
            "INTERNAL_ARRIVAL"
        ):

            r = ride_map[key]

            destination = r.end_station_id

            if kind == "EXTERNAL_ARRIVAL":

                counts["exogenous_in"] += 1

            else:

                if key not in in_transit:

                    raise AssertionError(
                        "内部归还没有成功借出的记录"
                    )

                in_transit.remove(key)

            counts["arrival_attempts"] += 1

            station_hour[
                (destination, now.hour)
            ]["arrival_attempts"] += 1

            # 有空桩，直接归还
            if (
                dock[destination]
                < capacity[destination]
                and not queue[destination]
            ):

                dock[destination] += 1

                counts["immediate_dock"] += 1

                station_hour[
                    (destination, now.hour)
                ]["immediate_dock"] += 1

            # 满桩，开始等待
            else:

                idx = len(wait_logs)

                wait_logs.append({

                    "scenario": scenario,
                    "ride_id": key,
                    "station_id": destination,

                    "arrival_at": str(now),
                    "exit_at": "",

                    "status": "waiting",

                    "observed_wait_minutes": None,

                    "_arrived": now
                })

                queue[destination].append(idx)

                counts["wait_occurrences"] += 1

                station_hour[
                    (destination, now.hour)
                ]["wait_occurrences"] += 1

                # 30分钟后安排超时事件
                schedule(
                    now + patience,
                    2,
                    "TIMEOUT",
                    idx
                )

            fill_waiting(destination, now)

        else:

            raise AssertionError(
                f"未知事件类型：{kind}"
            )

        # 每一事件后立即检查车辆守恒
        conservation()

    # ==================================================
    # 4. 午夜右删失处理
    # ==================================================

    for row in wait_logs:

        if row["status"] == "waiting":

            row["status"] = "censored_at_midnight"

            row["observed_wait_minutes"] = (
                end_day - row["_arrived"]
            ).total_seconds() / 60.0

        del row["_arrived"]

    pending = sum(
        len(q)
        for q in queue.values()
    )

    # ==================================================
    # 5. 账本验收
    # ==================================================

    assert (
        counts["requests"]
        ==
        counts["accepted"]
        + counts["rejected"]
    )

    assert (
        counts["arrival_attempts"]
        ==
        counts["immediate_dock"]
        + counts["wait_occurrences"]
    )

    assert (
        counts["wait_occurrences"]
        ==
        counts["wait_resolved"]
        + counts["timeout_external_exit"]
        + pending
    )

    for row in wait_logs:

        if (
            row["status"] == "censored_at_midnight"
            and row["observed_wait_minutes"]
            >= WAIT_LIMIT_MIN + TOL
        ):

            raise AssertionError(
                "存在应当超时却未执行的事件"
            )

        if (
            row["status"] == "docked_after_wait"
            and row["observed_wait_minutes"]
            > WAIT_LIMIT_MIN + TOL
        ):

            raise AssertionError(
                "存在超过等待上限的成功入桩"
            )

    # ==================================================
    # 6. 逐站逐小时数据
    # ==================================================

    station_hour_rows = []

    for station in ids:

        backlog = 0

        for hour in range(24):

            c = station_hour[
                (station, hour)
            ]

            backlog += (
                c["wait_occurrences"]
                - c["wait_resolved"]
                - c["timeout_external_exit"]
            )

            if backlog < 0:

                raise AssertionError(
                    "小时等待队列出现负数"
                )

            station_hour_rows.append({

                "scenario": scenario,
                "station_id": station,
                "hour": hour,

                "requests": c["requests"],
                "accepted": c["accepted"],
                "rejected": c["rejected"],

                "arrival_attempts":
                    c["arrival_attempts"],

                "immediate_dock":
                    c["immediate_dock"],

                "wait_occurrences":
                    c["wait_occurrences"],

                "wait_resolved":
                    c["wait_resolved"],

                "timeout_external_exit":
                    c["timeout_external_exit"],

                "queue_end": backlog
            })

        if backlog != len(queue[station]):

            raise AssertionError(
                "站点日末排队数不一致"
            )

    # ==================================================
    # 7. 服务指标
    # ==================================================

    observed_waits = np.array(
        [
            r["observed_wait_minutes"]
            for r in wait_logs
        ],
        dtype=float
    )

    resolved_waits = np.array(
        [
            r["observed_wait_minutes"]
            for r in wait_logs
            if r["status"] == "docked_after_wait"
        ],
        dtype=float
    )

    # 目标站点完成入桩
    successful_returns = (
        counts["immediate_dock"]
        + counts["wait_resolved"]
    )

    requests = counts["requests"]
    arrivals = counts["arrival_attempts"]

    summary = {

        "scenario": scenario,

        "initial_bikes": sum(initial.values()),

        "requests": requests,

        "borrow_accepted":
            counts["accepted"],

        "borrow_rejected":
            counts["rejected"],

        "borrow_success_rate": (
            counts["accepted"] / requests
            if requests else None
        ),

        "arrival_attempts": arrivals,

        "immediate_returns":
            counts["immediate_dock"],

        "wait_occurrences":
            counts["wait_occurrences"],

        "wait_resolved":
            counts["wait_resolved"],

        "timeout_external_exit":
            counts["timeout_external_exit"],

        "pending_at_midnight":
            pending,

        "target_station_docked":
            successful_returns,

        "target_station_docking_rate": (
            successful_returns / arrivals
            if arrivals else None
        ),

        "mean_resolved_wait_min": (
            float(resolved_waits.mean())
            if len(resolved_waits) else None
        ),

        "p95_resolved_wait_min": (
            float(np.percentile(
                resolved_waits, 95
            ))
            if len(resolved_waits) else None
        ),

        # 包含已经历的等待：
        # 已完成、已超时及日末右删失记录
        "observed_total_wait_min": (
            float(observed_waits.sum())
            if len(observed_waits) else 0.0
        ),

        "docked_bikes_at_midnight":
            sum(dock.values()),

        "internal_in_transit_at_midnight":
            len(in_transit),

        "exogenous_in":
            counts["exogenous_in"],

        "outbound_departed":
            counts["outbound_departed"],

        "cancelled_internal_arrivals_today":
            counts[
                "cancelled_internal_arrivals_today"
            ],

        "vehicle_conservation": "PASS"
    }

    return (
        summary,
        wait_logs,
        station_hour_rows
    )


# ==================================================
# 8. 主程序
# ==================================================

def main():

    print("=" * 65)
    print("Day 5-E：最大等待30分钟的事件仿真")
    print("=" * 65)

    (
        ids,
        capacity,
        inventories,
        trips,
        expected_starts,
        expected_ends,
        ignored
    ) = base.read_inputs()

    print(
        f"测试日={base.DATE}，"
        f"区域站点={len(ids)}，"
        f"订单={len(trips):,}"
    )

    print(
        f"观测借车={expected_starts:,}，"
        f"观测归还={expected_ends:,}"
    )

    # 冻结的Day5-B汇总
    old_file = (
        base.OUT_DIR
        / "day5_baseline_summary.csv"
    )

    if not old_file.is_file():

        raise FileNotFoundError(
            f"缺少冻结基线：{old_file}"
        )

    frozen = pd.read_csv(
        old_file
    ).set_index("scenario")

    summaries = []
    waiting_records = []
    all_station_hours = []

    # ==================================================
    # 8.1 四种库存情景
    # ==================================================

    for scenario in base.SCENARIOS:

        summary, waits, rows = replay(
            ids,
            capacity,
            inventories[scenario],
            trips,
            scenario
        )

        # 请求数量必须保持不变
        if summary["requests"] != expected_starts:

            raise AssertionError(
                f"{scenario}借车请求数错误"
            )

        # 归还事件必须能与历史记录对应
        if (
            summary["arrival_attempts"]
            + summary[
                "cancelled_internal_arrivals_today"
            ]
            != expected_ends
        ):

            raise AssertionError(
                f"{scenario}归还事件对账失败"
            )

        # 与原始冻结基线保持相同的输入请求
        old_requests = int(
            frozen.loc[
                scenario,
                "requests_from_area"
            ]
        )

        if summary["requests"] != old_requests:

            raise AssertionError(
                f"{scenario}请求量与冻结基线不一致"
            )

        summaries.append(summary)
        waiting_records.extend(waits)
        all_station_hours.extend(rows)

        print()
        print(
            f"情景：{scenario}"
        )

        print(
            "  借车拒绝：",
            summary["borrow_rejected"]
        )

        print(
            "  满桩等待发生：",
            summary["wait_occurrences"]
        )

        print(
            "  等待30分钟后外部退出：",
            summary["timeout_external_exit"]
        )

        print(
            "  日末未解决等待：",
            summary["pending_at_midnight"]
        )

        print(
            "  目标站归还完成率："
            f"{summary['target_station_docking_rate']:.2%}"
        )

        print(
            "  车辆守恒：PASS"
        )

    # ==================================================
    # 8.2 保存
    # ==================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(summaries).to_csv(
        OUT_DIR / "timeout_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(waiting_records).to_csv(
        OUT_DIR / "timeout_wait_log.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(all_station_hours).to_csv(
        OUT_DIR / "timeout_station_hour.csv",
        index=False,
        encoding="utf-8-sig"
    )

    audit = {

        "date": base.DATE,

        "max_wait_minutes":
            WAIT_LIMIT_MIN,

        "boundary_policy": (
            "超时车辆离开25站研究区域，"
            "不虚构替代归还站点。"
        ),

        "same_timestamp_order": (
            "先借车，再归还，最后处理超时；"
            "恰好30分钟时释放空桩可以成功入桩。"
        ),

        "limitations": [

            "历史已完成订单作为反事实请求输入。",

            "30分钟是预设的假设政策，"
            "不是利用测试日拟合的参数。",

            "外部退出不代表已知真实替代归还站点。",

            "午夜右删失不计作超时。",

            "尚未实施运营方主动调拨。",

            "未考虑车辆运输时延和卡车容量。"
        ],

        "results": summaries
    }

    (
        OUT_DIR / "timeout_audit.json"
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
        "PASS：全部情景通过事件账本与车辆守恒检查"
    )

    print(
        "输出目录：", OUT_DIR
    )


if __name__ == "__main__":
    main()
