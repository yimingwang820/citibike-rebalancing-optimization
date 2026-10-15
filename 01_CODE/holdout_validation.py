
# -*- coding: utf-8 -*-
"""
Day 6-C：冻结预调拨方案，跨日期事件仿真验证

验证：2026-04-22 至 2026-04-30
策略：沿用Day 6-A的80辆、6条调拨关系
基线：balanced_50，共853辆
等待：30分钟

逐日完成：
1. 原始ZIP订单提取
2. 与修复后的逐小时流量对账
3. 无调拨 / 预调拨事件回放
4. 车辆守恒验证
5. 工作日和周末分组汇总

不调用Gurobi，也不重新优化。
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json

import pandas as pd

import event_baseline as baseline_module
import timeout_simulation as timeout_module
import extract_trip_events as extract_module


ROOT = Path(r"D:\BikeSharing_OR_Project")

PLAN_DIR = (
    ROOT / "02_RESULTS"
    / "DAY6_PREPOSITION"
)

OUT_DIR = (
    ROOT / "02_RESULTS"
    / "DAY6_HOLDOUT"
)

DATES = [
    d.strftime("%Y-%m-%d")
    for d in pd.date_range(
        "2026-04-22",
        "2026-04-30"
    )
]

DATE_SET = set(DATES)

SEED = "5980.10"
SCENARIO = "balanced_50"

EXPECTED_BIKES = 853
EXPECTED_MOVES = 80
EXPECTED_PAIRS = 6


# ==================================================
# 1. 文件指纹
# ==================================================

def file_sha256(path):

    h = sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


# ==================================================
# 2. 读取冻结的训练期方案
# ==================================================

def load_frozen_plan():

    invfile = (
        PLAN_DIR / "initial_inventory_comparison.csv"
    )

    planfile = (
        PLAN_DIR / "training_preposition_plan.csv"
    )

    for p in (invfile, planfile):
        if not p.is_file():
            raise FileNotFoundError(p)

    inv = pd.read_csv(
        invfile,
        dtype={"station_id": str}
    )

    plan = pd.read_csv(
        planfile,
        dtype={
            "origin_id": str,
            "destination_id": str
        }
    )

    st = pd.read_csv(
        baseline_module.STATIONS_FILE,
        dtype={
            "station_id": str,
            "candidate_seed": str
        }
    )

    st = st.loc[
        st["candidate_seed"] == SEED
    ]

    raw_inv = pd.read_csv(
        baseline_module.INVENTORY_FILE,
        dtype={"station_id": str}
    )

    raw_inv = raw_inv.loc[
        raw_inv["scenario"] == SCENARIO
    ]

    ids = sorted(st["station_id"])

    if (
        len(ids) != 25
        or len(set(ids)) != 25
        or set(inv.station_id) != set(ids)
    ):
        raise ValueError(
            "冻结站点或库存不是同一25站集合"
        )

    for df in (st, raw_inv, inv):
        if df["station_id"].duplicated().any():
            raise ValueError(
                "存在重复站点ID"
            )

    if set(raw_inv.station_id) != set(ids):
        raise ValueError(
            "原始库存站点集合不一致"
        )

    old = {
        r.station_id: int(r.before)
        for r in inv.itertuples(index=False)
    }

    new = {
        r.station_id: int(r.after)
        for r in inv.itertuples(index=False)
    }

    capacity = {
        r.station_id: int(r.proxy_capacity)
        for r in st.itertuples(index=False)
    }

    reference = {
        r.station_id: int(r.initial_inventory)
        for r in raw_inv.itertuples(index=False)
    }

    if (
        old != reference
        or sum(old.values()) != EXPECTED_BIKES
        or sum(new.values()) != EXPECTED_BIKES
    ):
        raise AssertionError(
            "冻结库存未匹配853辆balanced_50"
        )

    for i in ids:

        if not (
            0 <= old[i] <= capacity[i]
            and 0 <= new[i] <= capacity[i]
        ):
            raise AssertionError(
                f"站点{i}库存超容量"
            )

    if (
        len(plan) != EXPECTED_PAIRS
        or int(plan["bikes"].sum()) != EXPECTED_MOVES
    ):
        raise AssertionError(
            "冻结方案不是80辆6条关系"
        )

    if plan[
        ["origin_id", "destination_id"]
    ].duplicated().any():
        raise AssertionError(
            "冻结调拨存在重复关系"
        )

    if not (
        plan.origin_id.isin(ids).all()
        and plan.destination_id.isin(ids).all()
    ):
        raise AssertionError(
            "调拨出现区域外站点"
        )

    if (
        plan.origin_id == plan.destination_id
    ).any():
        raise AssertionError(
            "存在自环调拨"
        )

    if not (
        plan.bikes.gt(0).all()
        and (plan.bikes % 1 == 0).all()
    ):
        raise AssertionError(
            "出现无效调拨数量"
        )

    flow_in = (
        plan.groupby("destination_id")["bikes"]
        .sum().to_dict()
    )

    flow_out = (
        plan.groupby("origin_id")["bikes"]
        .sum().to_dict()
    )

    for i in ids:

        delta = (
            flow_in.get(i, 0)
            - flow_out.get(i, 0)
        )

        if new[i] - old[i] != int(delta):
            raise AssertionError(
                f"站点{i}调拨账本不符"
            )

    # 合成模型成本，并非实际货币
    nominal_cost = (
        float(plan.transport_cost.sum())
        + 3.0 * len(plan)
    )

    fingerprints = {
        "inventory_csv_sha256":
            file_sha256(invfile),

        "plan_csv_sha256":
            file_sha256(planfile)
    }

    return (
        ids,
        capacity,
        old,
        new,
        nominal_cost,
        fingerprints
    )


# ==================================================
# 3. 从原始ZIP提取保留日期订单
# ==================================================

def extract_holdout(ids):

    selected = set(ids)

    mapping = pd.read_csv(
        extract_module.MAPPING_FILE,
        dtype=str
    )

    if mapping.raw_station_id.duplicated().any():
        raise ValueError(
            "站点ID映射重复"
        )

    aliases = dict(zip(
        mapping.raw_station_id,
        mapping.canonical_station_id
    ))

    parts = []
    seen = set()

    invalid_count = 0
    duplicate_count = 0
    rows_scanned = 0

    with ZipFile(
        extract_module.ZIP_FILE
    ) as z:

        filenames = sorted(
            n for n in z.namelist()
            if n.lower().endswith(".csv")
            and not n.startswith("__MACOSX/")
        )

        if not filenames:
            raise RuntimeError(
                "ZIP没有CSV"
            )

        for fname in filenames:

            with z.open(fname) as fp:

                reader = pd.read_csv(
                    fp,
                    usecols=extract_module.COLUMNS,
                    chunksize=100000,
                    low_memory=False,
                    dtype={
                        "ride_id": str,
                        "start_station_id": str,
                        "end_station_id": str
                    }
                )

                for chunk in reader:

                    rows_scanned += len(chunk)

                    raw_start = (
                        chunk.start_station_id
                        .fillna("")
                        .str.strip()
                    )

                    raw_end = (
                        chunk.end_station_id
                        .fillna("")
                        .str.strip()
                    )

                    start_id = (
                        raw_start.map(aliases)
                        .fillna(raw_start)
                    )

                    end_id = (
                        raw_end.map(aliases)
                        .fillna(raw_end)
                    )

                    local = (
                        start_id.isin(selected)
                        | end_id.isin(selected)
                    )

                    matching_date = (
                        chunk.started_at
                        .astype("string")
                        .str[:10]
                        .isin(DATE_SET)
                        |
                        chunk.ended_at
                        .astype("string")
                        .str[:10]
                        .isin(DATE_SET)
                    )

                    take = local & matching_date

                    if not take.any():
                        continue

                    part = chunk.loc[take].copy()

                    part["raw_start_station_id"] = (
                        raw_start.loc[take].to_numpy()
                    )

                    part["raw_end_station_id"] = (
                        raw_end.loc[take].to_numpy()
                    )

                    part["start_station_id"] = (
                        start_id.loc[take].to_numpy()
                    )

                    part["end_station_id"] = (
                        end_id.loc[take].to_numpy()
                    )

                    st = pd.to_datetime(
                        part.started_at,
                        errors="coerce"
                    )

                    en = pd.to_datetime(
                        part.ended_at,
                        errors="coerce"
                    )

                    seconds = (
                        en - st
                    ).dt.total_seconds()

                    ride_id = (
                        part.ride_id.fillna("")
                        .astype(str)
                        .str.strip()
                    )

                    valid = (
                        st.notna()
                        & en.notna()
                        & seconds.gt(0)
                        & seconds.le(86400)
                        & st.dt.strftime("%Y-%m")
                        .eq("2026-04")
                        & ride_id.ne("")
                        & part.start_station_id.ne("")
                        & part.end_station_id.ne("")
                    )

                    for col in (
                        "start_lat",
                        "start_lng",
                        "end_lat",
                        "end_lng"
                    ):
                        part[col] = pd.to_numeric(
                            part[col],
                            errors="coerce"
                        )

                    valid &= (
                        part.start_lat.between(-90, 90)
                        & part.end_lat.between(-90, 90)
                        & part.start_lng.between(-180, 180)
                        & part.end_lng.between(-180, 180)
                    )

                    duplicated = (
                        ride_id.duplicated(
                            keep="first"
                        )
                        | ride_id.isin(seen)
                    )

                    duplicate_count += int(
                        (duplicated & valid).sum()
                    )

                    seen.update(
                        ride_id[
                            ride_id.ne("")
                        ].tolist()
                    )

                    valid &= ~duplicated

                    invalid_count += int(
                        (~valid).sum()
                    )

                    if not valid.any():
                        continue

                    part = part.loc[valid].copy()

                    part["start_date"] = (
                        st.loc[valid]
                        .dt.strftime("%Y-%m-%d")
                        .to_numpy()
                    )

                    part["end_date"] = (
                        en.loc[valid]
                        .dt.strftime("%Y-%m-%d")
                        .to_numpy()
                    )

                    part["start_hour"] = (
                        st.loc[valid]
                        .dt.hour.to_numpy()
                    )

                    part["end_hour"] = (
                        en.loc[valid]
                        .dt.hour.to_numpy()
                    )

                    # 事件仿真器需要Timestamp对象
                    part["started_at"] = (
                        st.loc[valid].to_numpy()
                    )

                    part["ended_at"] = (
                        en.loc[valid].to_numpy()
                    )

                    parts.append(part)

    if not parts:
        raise RuntimeError(
            "未提取到保留日订单"
        )

    trips = pd.concat(
        parts,
        ignore_index=True
    )

    if trips.ride_id.duplicated().any():
        raise AssertionError(
            "提取后ride_id重复"
        )

    trips = trips.sort_values(
        ["started_at", "ride_id"]
    ).reset_index(drop=True)

    statistics = {
        "zip_scanned_rows":
            rows_scanned,

        "invalid_or_duplicate_relevant_rows":
            invalid_count,

        "duplicate_relevant_ids":
            duplicate_count,

        "trip_pairs_extracted":
            len(trips)
    }

    return trips, statistics


# ==================================================
# 4. 与修复后的小时数据逐日对账
# ==================================================

def reconcile(trips, ids):

    selected = set(ids)

    ref_parts = []

    reader = pd.read_csv(
        extract_module.FLOW_FILE,
        chunksize=250000,
        dtype={
            "station_id": str,
            "date": str,
            "type": str
        }
    )

    for chunk in reader:

        keep = (
            chunk.date.isin(DATE_SET)
            & chunk.station_id.isin(selected)
        )

        if keep.any():

            ref_parts.append(
                chunk.loc[
                    keep,
                    [
                        "date",
                        "station_id",
                        "hour",
                        "type",
                        "count"
                    ]
                ]
            )

    if not ref_parts:
        raise RuntimeError(
            "修复后的小时表没有测试日期"
        )

    reference = (
        pd.concat(ref_parts)
        .groupby(
            [
                "date",
                "station_id",
                "hour",
                "type"
            ]
        )["count"].sum()
    )

    starts = trips.loc[
        trips.start_date.isin(DATE_SET)
        & trips.start_station_id.isin(selected),
        [
            "start_date",
            "start_station_id",
            "start_hour"
        ]
    ].copy()

    starts.columns = [
        "date",
        "station_id",
        "hour"
    ]

    starts["type"] = "start"

    ends = trips.loc[
        trips.end_date.isin(DATE_SET)
        & trips.end_station_id.isin(selected),
        [
            "end_date",
            "end_station_id",
            "end_hour"
        ]
    ].copy()

    ends.columns = [
        "date",
        "station_id",
        "hour"
    ]

    ends["type"] = "end"

    observed = (
        pd.concat([starts, ends])
        .groupby(
            [
                "date",
                "station_id",
                "hour",
                "type"
            ]
        ).size()
    )

    index = pd.MultiIndex.from_product(
        [
            DATES,
            ids,
            list(range(24)),
            ["start", "end"]
        ],
        names=[
            "date",
            "station_id",
            "hour",
            "type"
        ]
    )

    full = pd.DataFrame({
        "extracted":
            observed.reindex(
                index, fill_value=0
            ).astype(int),

        "reference":
            reference.reindex(
                index, fill_value=0
            ).astype(int)
    })

    full["difference"] = (
        full.extracted
        - full.reference
    )

    rec = full.reset_index()

    failures = rec.loc[
        rec.difference.ne(0)
    ]

    return rec, failures


# ==================================================
# 5. 多日期汇总函数
# ==================================================

def make_group(rows, title, cost):

    count = len(rows)

    base_rej = sum(
        x["base_rejected"] for x in rows
    )

    plan_rej = sum(
        x["plan_rejected"] for x in rows
    )

    base_timeout = sum(
        x["base_timeouts"] for x in rows
    )

    plan_timeout = sum(
        x["plan_timeouts"] for x in rows
    )

    b_arrivals = sum(
        x["base_arrivals"] for x in rows
    )

    p_arrivals = sum(
        x["plan_arrivals"] for x in rows
    )

    return {
        "group": title,
        "days": count,

        "requests": sum(
            x["requests"] for x in rows
        ),

        "base_rejected": base_rej,
        "plan_rejected": plan_rej,

        "change_rejected":
            plan_rej - base_rej,

        "base_timeout": base_timeout,
        "plan_timeout": plan_timeout,

        "change_timeout":
            plan_timeout - base_timeout,

        "base_return_completion_rate": (
            sum(x["base_docked"] for x in rows)
            / b_arrivals
            if b_arrivals else None
        ),

        "plan_return_completion_rate": (
            sum(x["plan_docked"] for x in rows)
            / p_arrivals
            if p_arrivals else None
        ),

        "base_observed_wait_min": round(
            sum(
                x["base_wait_min"] for x in rows
            ),
            2
        ),

        "plan_observed_wait_min": round(
            sum(
                x["plan_wait_min"] for x in rows
            ),
            2
        ),

        "days_timeout_better": sum(
            x["change_timeouts"] < 0
            for x in rows
        ),

        "days_timeout_equal": sum(
            x["change_timeouts"] == 0
            for x in rows
        ),

        "days_timeout_worse": sum(
            x["change_timeouts"] > 0
            for x in rows
        ),

        "nominal_total_preposition_cost_units":
            round(count * cost, 4)
    }


# ==================================================
# 6. 主程序
# ==================================================

def main():

    print("=" * 64)
    print(
        "Day 6-C：冻结预调拨策略 · "
        "4月22–30日跨日期验证"
    )
    print("=" * 64)

    if timeout_module.WAIT_LIMIT_MIN != 30:
        raise AssertionError(
            "等待政策已不是30分钟，拒绝比较"
        )

    # ----------------------------------------------
    # 6.1 冻结调拨方案
    # ----------------------------------------------

    (
        ids,
        capacity,
        before,
        after,
        cost,
        fingerprints
    ) = load_frozen_plan()

    print(
        "[1/4] 冻结方案通过："
        f"25站，853辆，"
        f"80辆预调拨，6条关系，"
        f"成本单位/日={cost:.3f}"
    )

    # ----------------------------------------------
    # 6.2 提取所有测试日订单
    # ----------------------------------------------

    trips, stats = extract_holdout(ids)

    print(
        "[2/4] 从原始ZIP提取相关订单："
        f"{len(trips):,} 条"
    )

    # ----------------------------------------------
    # 6.3 数据对账
    # ----------------------------------------------

    rec, failures = reconcile(
        trips, ids
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rec.to_csv(
        OUT_DIR / "holdout_hourly_reconciliation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if len(failures):

        failures.to_csv(
            OUT_DIR / "holdout_mismatches.csv",
            index=False,
            encoding="utf-8-sig"
        )

        raise AssertionError(
            f"小时事件对账失败：{len(failures)}个"
            "站点-小时-事件类型；"
            "先修复数据，不可进行正式评估。"
        )

    print(
        "[3/4] PASS：9天全部站点的小时借还"
        "事件与修复后汇总一致"
    )

    # ----------------------------------------------
    # 6.4 逐日事件仿真
    # ----------------------------------------------

    records = []
    station_hour = []

    old_date = baseline_module.DATE

    try:

        for date in DATES:

            # 仿真器原先固定使用4月21日。
            # 顺序回放时临时设置日期，
            # 回放结束后恢复，不修改任何源文件。
            baseline_module.DATE = date

            starts = (
                trips.start_date.eq(date)
                & trips.start_station_id.isin(ids)
            )

            ends = (
                trips.end_date.eq(date)
                & trips.end_station_id.isin(ids)
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

            if day_trips.empty:
                raise RuntimeError(
                    f"{date}没有提取到订单"
                )

            # 原始853辆，无调拨
            base, _, base_hour = (
                timeout_module.replay(
                    ids,
                    capacity,
                    before,
                    day_trips,
                    "baseline_balanced_50"
                )
            )

            # 同一日期、同一订单，
            # 使用冻结后的调拨库存
            plan, _, plan_hour = (
                timeout_module.replay(
                    ids,
                    capacity,
                    after,
                    day_trips,
                    "training_preposition"
                )
            )

            # --------------------------------------
            # 借还事件及车辆守恒验收
            # --------------------------------------

            for label, result in (
                ("baseline", base),
                ("plan", plan)
            ):

                if result["requests"] != expected_starts:
                    raise AssertionError(
                        f"{date} {label}"
                        "借车事件数未对账"
                    )

                actual_returns = (
                    result["arrival_attempts"]
                    + result[
                        "cancelled_internal_arrivals_today"
                    ]
                )

                if actual_returns != expected_ends:
                    raise AssertionError(
                        f"{date} {label}"
                        "归还事件数未对账"
                    )

                if (
                    result["vehicle_conservation"]
                    != "PASS"
                ):
                    raise AssertionError(
                        f"{date} {label}"
                        "车辆不守恒"
                    )

            for row in base_hour + plan_hour:

                station_hour.append({
                    "date": date,
                    **row
                })

            weekday = (
                "weekday"
                if pd.Timestamp(date).weekday() < 5
                else "weekend"
            )

            # --------------------------------------
            # 每日评价指标
            # --------------------------------------

            row = {

                "date": date,
                "day_type": weekday,

                "requests": expected_starts,

                "historical_arrival_events":
                    expected_ends,

                "base_rejected":
                    int(base["borrow_rejected"]),

                "plan_rejected":
                    int(plan["borrow_rejected"]),

                "change_rejected":
                    int(
                        plan["borrow_rejected"]
                        - base["borrow_rejected"]
                    ),

                "base_timeouts":
                    int(base["timeout_external_exit"]),

                "plan_timeouts":
                    int(plan["timeout_external_exit"]),

                "change_timeouts":
                    int(
                        plan["timeout_external_exit"]
                        - base["timeout_external_exit"]
                    ),

                "base_arrivals":
                    int(base["arrival_attempts"]),

                "plan_arrivals":
                    int(plan["arrival_attempts"]),

                "base_docked":
                    int(base["target_station_docked"]),

                "plan_docked":
                    int(plan["target_station_docked"]),

                "base_dock_rate":
                    float(
                        base[
                            "target_station_docking_rate"
                        ]
                    ),

                "plan_dock_rate":
                    float(
                        plan[
                            "target_station_docking_rate"
                        ]
                    ),

                "base_wait_min":
                    float(
                        base["observed_total_wait_min"]
                    ),

                "plan_wait_min":
                    float(
                        plan["observed_total_wait_min"]
                    ),

                "base_pending":
                    int(
                        base["pending_at_midnight"]
                    ),

                "plan_pending":
                    int(
                        plan["pending_at_midnight"]
                    ),

                "move_bikes": EXPECTED_MOVES,

                "synthetic_preposition_cost_units":
                    round(cost, 4)
            }

            records.append(row)

            print(
                f"  {date} {weekday:7} "
                "借车拒绝 "
                f"{row['base_rejected']}"
                "→"
                f"{row['plan_rejected']}；"
                "归还超时 "
                f"{row['base_timeouts']}"
                "→"
                f"{row['plan_timeouts']}；"
                "净变化 "
                f"{row['change_timeouts']:+d}"
            )

    finally:

        baseline_module.DATE = old_date

    # ----------------------------------------------
    # 6.5 工作日与周末分组
    # ----------------------------------------------

    aggregate = [
        make_group(
            records,
            "all",
            cost
        ),

        make_group(
            [
                x for x in records
                if x["day_type"] == "weekday"
            ],
            "weekday",
            cost
        ),

        make_group(
            [
                x for x in records
                if x["day_type"] == "weekend"
            ],
            "weekend",
            cost
        )
    ]

    # ----------------------------------------------
    # 6.6 保存结果
    # ----------------------------------------------

    pd.DataFrame(records).to_csv(
        OUT_DIR / "holdout_daily_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(aggregate).to_csv(
        OUT_DIR / "holdout_group_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(station_hour).to_csv(
        OUT_DIR / "holdout_station_hour_detail.csv",
        index=False,
        encoding="utf-8-sig"
    )

    audit = {

        "training": (
            "2026-04-01 to 2026-04-20, "
            "14 weekdays, 00:00-11:59 profile"
        ),

        "development_date_excluded":
            "2026-04-21",

        "holdout_dates": DATES,

        "policy": (
            "Same frozen midnight preposition: "
            "80 bikes, 6 arcs, on each independent day"
        ),

        "scenario": SCENARIO,

        "wait_limit_min":
            timeout_module.WAIT_LIMIT_MIN,

        "same_midnight_initial_stock_each_day":
            True,

        "do_not_interpret_as_continuous_9_day_simulation":
            True,

        "independent_day_cross_midnight_ride_policy":
            (
                "Prior-day returns treated as "
                "exogenous arrivals on each new day"
            ),

        "hourly_reconciliation_mismatches": 0,

        "source_plan_hashes": fingerprints,

        "extraction": stats,

        "aggregate": aggregate,

        "limitations": [

            "每个测试日均重置为相同的853辆"
            "合成初始库存，非连续多日运营。",

            "跨日前一日出发、当日归还订单"
            "记为外生到达，可能与合成"
            "初始库存假设不完全匹配。",

            "历史已完成订单作为反事实"
            "需求输入，拒绝及超时属于模拟指标。",

            "跨边界流入流出及30分钟超时退出"
            "按区域开放系统规则处理。",

            "预调拨假设在午夜前瞬时完成，"
            "未考虑卡车运力和运输时延。",

            "冻结训练策略，不使用4月22–30日"
            "数据重新优化。",

            "目标站归还完成率分母可能因"
            "内部借车拒绝而略有差异。",

            "运输成本为合成成本单位，"
            "非真实运营费用。"
        ]
    }

    (
        OUT_DIR / "holdout_audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # ----------------------------------------------
    # 6.7 终端汇总
    # ----------------------------------------------

    print(
        "[4/4] 跨日期评估完成，"
        "9天车辆守恒全部PASS"
    )

    print()
    print("========== 跨日期汇总 ==========")

    for a in aggregate:

        print(
            f"{a['group']:<7} "
            f"{a['days']}天 | "
            "借车拒绝 "
            f"{a['base_rejected']}"
            "→"
            f"{a['plan_rejected']} "
            f"({a['change_rejected']:+d}) | "
            "归还超时 "
            f"{a['base_timeout']}"
            "→"
            f"{a['plan_timeout']} "
            f"({a['change_timeout']:+d}) | "
            "超时改善/持平/恶化 "
            f"{a['days_timeout_better']}/"
            f"{a['days_timeout_equal']}/"
            f"{a['days_timeout_worse']}"
        )

    print()
    print("输出目录：", OUT_DIR)


if __name__ == "__main__":
    main()
