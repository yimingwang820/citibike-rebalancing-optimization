
# -*- coding: utf-8 -*-
"""
Day 4-C：多周期调拨方案可行性审计

检查：
1. 79辆车的逐小时调拨分布
2. 5个有向关系-小时的具体方案
3. 每个站点的库存守恒
4. 在“先借后还”保守时间顺序下的供给缺口

直接运行，无需额外参数。
"""

from pathlib import Path
import pandas as pd


from project_paths import ROOT

RESULTS = (
    ROOT / "02_RESULTS"
    / "DAY4_MULTIPERIOD_MILP"
)

STATIONS_FILE = (
    ROOT / "02_RESULTS"
    / "DAY2_A_FIXED"
    / "day2_selected_candidates.csv"
)


def main():

    files = {
        "hour": RESULTS / "day4_milp_by_hour.csv",
        "station": RESULTS / "day4_milp_stock_by_station.csv",
        "transfer": RESULTS / "day4_milp_transfers.csv"
    }

    for path in [*files.values(), STATIONS_FILE]:
        if not path.is_file():
            raise FileNotFoundError(path)

    hour = pd.read_csv(files["hour"])

    station = pd.read_csv(
        files["station"],
        dtype={"station_id": str}
    )

    transfer = pd.read_csv(
        files["transfer"],
        dtype={
            "origin_id": str,
            "destination_id": str
        }
    )

    meta = pd.read_csv(
        STATIONS_FILE,
        dtype={
            "station_id": str,
            "candidate_seed": str
        }
    )

    meta = meta.loc[
        meta["candidate_seed"] == "5980.10",
        ["station_id", "station_name", "proxy_capacity"]
    ].copy()

    # ==========================================
    # 1. 逐小时汇总
    # ==========================================

    print("\n========== 逐小时结果 ==========")

    for model_name, part in hour.groupby("model"):

        print(f"\n模型：{model_name}")

        print(
            part[
                [
                    "hour",
                    "stock_begin",
                    "observed_departures",
                    "observed_arrivals",
                    "external_in",
                    "external_out",
                    "moved_bikes",
                    "stock_end"
                ]
            ].to_string(index=False)
        )

    # ==========================================
    # 2. 显示全部调拨关系
    # ==========================================

    print("\n========== 调拨方案 ==========")

    if not transfer.empty:

        named = transfer.merge(
            meta[
                ["station_id", "station_name"]
            ].rename(
                columns={
                    "station_id": "origin_id",
                    "station_name": "origin_name"
                }
            ),
            on="origin_id",
            how="left",
            validate="many_to_one"
        )

        named = named.merge(
            meta[
                ["station_id", "station_name"]
            ].rename(
                columns={
                    "station_id": "destination_id",
                    "station_name": "destination_name"
                }
            ),
            on="destination_id",
            how="left",
            validate="many_to_one"
        )

        named = named.sort_values(
            ["hour", "origin_id", "destination_id"]
        )

        print(
            named[
                [
                    "hour",
                    "origin_id",
                    "destination_id",
                    "bikes",
                    "distance_km",
                    "origin_name",
                    "destination_name"
                ]
            ].to_string(index=False)
        )

        print(
            "\n总调拨车辆：",
            int(transfer["bikes"].sum())
        )

        print(
            "启用关系-小时数：",
            len(transfer)
        )

    else:
        print("没有调拨记录。")

    # ==========================================
    # 3. 汇总每站每小时调入、调出
    # ==========================================

    incoming = (
        transfer.groupby(
            ["hour", "destination_id"],
            as_index=False
        )["bikes"]
        .sum()
        .rename(
            columns={
                "destination_id": "station_id",
                "bikes": "transfer_in"
            }
        )
    )

    outgoing = (
        transfer.groupby(
            ["hour", "origin_id"],
            as_index=False
        )["bikes"]
        .sum()
        .rename(
            columns={
                "origin_id": "station_id",
                "bikes": "transfer_out"
            }
        )
    )

    reviews = []

    for model_name, part in station.groupby("model"):

        work = part.copy()

        if model_name == "with_rebalancing":

            work = work.merge(
                incoming,
                on=["hour", "station_id"],
                how="left"
            ).merge(
                outgoing,
                on=["hour", "station_id"],
                how="left"
            )

            work[
                ["transfer_in", "transfer_out"]
            ] = work[
                ["transfer_in", "transfer_out"]
            ].fillna(0)

        else:

            work["transfer_in"] = 0.0
            work["transfer_out"] = 0.0

        # 调拨后库存
        work["stock_after_transfer"] = (
            work["stock_begin"]
            + work["transfer_in"]
            - work["transfer_out"]
        )

        # 假设外部补车在借车前可用
        work["available_before_returns"] = (
            work["stock_after_transfer"]
            + work["external_in"]
        )

        # 假设本小时借车先于全部归还
        # 此时可能出现的保守供给缺口
        work["pre_return_gap"] = (
            work["observed_departures"]
            - work["available_before_returns"]
        ).clip(lower=0)

        # 原模型库存守恒的数值残差
        work["balance_residual"] = (
            work["stock_end"]
            - work["stock_begin"]
            - work["transfer_in"]
            + work["transfer_out"]
            - work["observed_arrivals"]
            + work["observed_departures"]
            - work["external_in"]
            + work["external_out"]
        )

        reviews.append(work)

    review = pd.concat(
        reviews, ignore_index=True
    )

    review = review.merge(
        meta,
        on="station_id",
        how="left",
        validate="many_to_one"
    )

    # ==========================================
    # 4. 核查库存守恒
    # ==========================================

    max_residual = (
        review["balance_residual"].abs().max()
    )

    print("\n========== 库存守恒 ==========")

    print(
        f"最大库存守恒残差：{max_residual:.10f}"
    )

    if max_residual > 1e-5:
        raise AssertionError(
            "出现库存守恒异常"
        )

    print("PASS：全部站点小时库存守恒。")

    # ==========================================
    # 5. 先借后还的保守情景审计
    # ==========================================

    print("\n========== 小时内时序风险 ==========")

    for model_name, part in review.groupby("model"):

        count = int(
            (part["pre_return_gap"] > 1e-6).sum()
        )

        total = part["pre_return_gap"].sum()

        print(
            f"{model_name}: "
            f"风险站点-小时数={count}, "
            f"保守缺口代理量={total:.3f}"
        )

    flagged = review.loc[
        review["pre_return_gap"] > 1e-6
    ].sort_values(
        "pre_return_gap",
        ascending=False
    )

    if not flagged.empty:

        print("\n保守缺口最大的前10条记录：")

        print(
            flagged[
                [
                    "model",
                    "hour",
                    "station_id",
                    "station_name",
                    "observed_departures",
                    "available_before_returns",
                    "pre_return_gap"
                ]
            ].head(10).to_string(index=False)
        )

    # ==========================================
    # 6. 保存审计结果
    # ==========================================

    output_file = (
        RESULTS / "day4_feasibility_review.csv"
    )

    review.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n审计完成。")
    print("结果文件：", output_file)

    print(
        "\n注意：保守缺口表示在"
        "“本小时全部借车先于归还”假设下的"
        "供给不足，并不等于真实历史缺车。"
    )


if __name__ == "__main__":
    main()
