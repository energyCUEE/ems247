"""
One-day rolling-horizon EMS with a soft green-purchase budget.

One run = one formulation (`mode` in the config: prescient, deterministic or
stochastic), swept over n_RE_days (targeted CF days per horizon). For each
planning date:
  1. Soft green budget = green_purchase_volume_factor * forecasted load energy
     over the horizon; exceeding it costs weight_green_slack per kWh.
  2. Solve the horizon problem.
  3. After WARMUP_DAYS, fix the first day's non-green status by an
     exponentially weighted vote of today's solution and the earlier plans for
     that same day, then re-solve.
  4. Execute the first day; its end SoC becomes the next initial SoC.
"""
import os
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
from src.re100_sp import BatteryParams, MeasurementData, ForecastData, RE100EMS
from src.copula import DailyCopulaErrorModel
warnings.filterwarnings("ignore")

WARMUP_DAYS = 6            # use the unfixed solution until enough history exists
LOOKBACK_N = 7             # number of votes (today + previous 6 plans)
DECAY_RATE = 0.1
CONSENSUS_THRESHOLD = 0.6
VOTE_WEIGHTS = np.exp(-DECAY_RATE * np.arange(1, LOOKBACK_N + 1))
VOTE_WEIGHTS /= VOTE_WEIGHTS.sum()


def main(arg):
    with open(arg.optimization_configs_path, "r") as f:
        cfg = json.load(f)
    with open(cfg["battery_configs_path"], "r") as f:
        battery_configs = json.load(f)

    mode = cfg["mode"]
    n_points_1day = cfg["n_points_1day"]
    n_horizon_days = cfg.get("n_horizon_days", 7)
    resolution_in_hour = 24 / n_points_1day
    factor = cfg.get("green_purchase_volume_factor")  # null -> no green budget
    weight_green_slack = cfg.get("weight_green_slack", 20)

    full_solar_df = pd.read_csv(cfg["solar_measurement_df_path"], parse_dates=['Datetime'])
    full_load_df = pd.read_csv(cfg["load_measurement_df_path"], parse_dates=['Datetime']).rename(columns={'L': 'PL'})
    measurement_data = MeasurementData(full_load_df=full_load_df, full_solar_df=full_solar_df)
    price_df = pd.read_csv(cfg["price_df_path"])
    solar_exp_df = pd.read_csv(cfg["solar_experiment_df_path"], parse_dates=['Date']).drop(columns=['y'])
    load_exp_df = pd.read_csv(cfg["load_experiment_df_path"], parse_dates=['Date']).drop(columns=['y'])
    forecast_data = ForecastData(exp_ir_df=solar_exp_df, exp_load_df=load_exp_df)
    copula_model = None
    if mode == "stochastic":
        copula_model = DailyCopulaErrorModel(pv_pipeline_path=cfg["pv_copula_pipeline_path"],
                                             load_pipeline_path=cfg["load_copula_pipeline_path"])

    start_planning_date = arg.start_planning_date or cfg["start_planning_date"]
    end_planning_date = arg.end_planning_date or cfg["end_planning_date"]
    planning_dates = pd.date_range(start=start_planning_date, end=end_planning_date, freq='D').strftime('%Y-%m-%d').tolist()
    os.makedirs(arg.saved_path, exist_ok=True)

    for n_RE_days in arg.n_RE_days:
        full_path = os.path.join(arg.saved_path, f"collected_rolling_result_{n_RE_days}REdays.csv")
        one_day_path = os.path.join(arg.saved_path, f"one_day_rolling_result_{n_RE_days}REdays.csv")
        for path in (full_path, one_day_path):
            if os.path.exists(path):
                os.remove(path)

        init_soc = battery_configs["init_socs"][0]
        plan_history = []  # executed u_normXnongreen_block per planning date
        t_start = time.time()
        for idx, planning_date in enumerate(planning_dates):
            if factor is None:
                green_quota, weight_slack = [np.inf], None
            else:
                forecasted_PL = forecast_data.get_forecasted_load(planning_date)
                green_quota, weight_slack = [factor * forecasted_PL.sum() * resolution_in_hour], weight_green_slack

            ems_kwargs = dict(
                battery=BatteryParams(**{**battery_configs, "init_socs": [init_soc]}),
                measurement_data=measurement_data,
                forecast_data=forecast_data,
                planning_date=planning_date,
                n_points_1day=n_points_1day,
                n_horizon_days=n_horizon_days,
                CFE_block_size=n_points_1day,  # one CFE block = one day
                n_CFE_blocks=n_RE_days,
                PV_cap_original=cfg["PV_cap_original"],
                PV_cap_target=cfg["Pv_cap_target"],
                price_df=price_df,
                terminal_socs=cfg["terminal_socs"],
                n_green_sources=1,
                green_allocation_quota=green_quota,
                weight_green_slack=weight_slack,
                mode=mode,
            )
            if mode == "stochastic":
                ems_kwargs.update(copula_model=copula_model, n_realizations=cfg["n_realizations"],
                                  seed=cfg.get("seed", 42) + idx + 1)

            t0 = time.time()
            prob, result_df, fval, exit_flag = RE100EMS(**ems_kwargs).optimize(solver=cp.GUROBI)
            plan = prob.var_dict["u_normXnongreen_block"].value.reshape(-1)

            fixed_status = None
            if idx >= WARMUP_DAYS:
                votes = [plan[0]] + [plan_history[idx - lead][lead] for lead in range(1, LOOKBACK_N)]
                fixed_status = int(VOTE_WEIGHTS @ votes > CONSENSUS_THRESHOLD)
                prob, result_df, fval, exit_flag = RE100EMS(**ems_kwargs, fixed_1st_day_nongreen_status=fixed_status).optimize(solver=cp.GUROBI)
                plan = prob.var_dict["u_normXnongreen_block"].value.reshape(-1)
            plan_history.append(plan)
            init_soc = float(prob.var_dict["soc"].value[n_points_1day, 0])

            # Real-time recourse on measured data -> DA/RT costs
            result_df = RE100EMS.adjust_RT_results_df(result_df, n_points_1hour=n_points_1day // 24, mode=mode)
            slack_green = prob.var_dict.get("slack_green")
            result_df["n_RE_days"] = n_RE_days
            result_df["planning_date"] = planning_date
            result_df["fixed_1st_day_nongreen_status"] = fixed_status
            result_df["budget_kwh"] = green_quota[0]
            result_df["slack_green_kwh"] = float(slack_green.value[0]) if slack_green is not None else 0.0
            result_df.to_csv(full_path, mode="a", header=not os.path.exists(full_path), index=False)
            result_df.iloc[:n_points_1day].to_csv(one_day_path, mode="a", header=not os.path.exists(one_day_path), index=False)
            print(f"[{mode}, {n_RE_days} RE days] {planning_date} ({idx + 1}/{len(planning_dates)}): "
                  f"fval={fval:.2f}, status={exit_flag}, fixed_1st_day={fixed_status}, "
                  f"next SoC={init_soc:.2f}, time={time.time() - t0:.1f}s", flush=True)

        one_day_df = pd.read_csv(one_day_path)
        print(f"[{mode}, {n_RE_days} RE days] done in {time.time() - t_start:.1f}s, "
              f"total net cost = {one_day_df['net_cost'].sum():.2f} THB, "
              f"green = {one_day_df[['Pgreen_DA', 'Pgreen_RT']].sum().sum() * resolution_in_hour:.1f} kWh, "
              f"non-green = {one_day_df[['Pnongreen_DA', 'Pnongreen_RT']].sum().sum() * resolution_in_hour:.1f} kWh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization_configs_path", type=str, default="./configs/stochastic_rolling_configs.json", help="path to optimization configs")
    parser.add_argument("--saved_path", type=str, default="./results/stochastic_rolling", help="path to save results")
    parser.add_argument("--n_RE_days", type=int, nargs="+", default=list(range(1, 8)), help="targeted CF days per horizon to sweep")
    parser.add_argument("--start_planning_date", type=str, default=None, help="override the config's start_planning_date")
    parser.add_argument("--end_planning_date", type=str, default=None, help="override the config's end_planning_date")
    args = parser.parse_args()
    main(args)
