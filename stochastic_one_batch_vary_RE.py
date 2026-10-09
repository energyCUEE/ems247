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
from src.plots import plot_one_batch_one_green_operation, plot_REdays_net_cost
warnings.filterwarnings("ignore")


def main(arg):
    with open(arg.optimization_configs_path, "r") as f:
        optimization_configs = json.load(f)
    with open(optimization_configs["battery_configs_path"], "r") as f:
        battery_configs = json.load(f)

    battery = BatteryParams(**battery_configs)
    planning_date = arg.planning_date
    solve_mode = arg.solve_mode  # 'optimize' or 'plot'

    # read measurement and forecast data
    full_solar_df = pd.read_csv(optimization_configs["solar_measurement_df_path"], parse_dates=['Datetime'])
    full_load_df = pd.read_csv(optimization_configs["load_measurement_df_path"], parse_dates=['Datetime']).rename(columns={'L': 'PL'})
    measurement_data = MeasurementData(full_load_df=full_load_df, full_solar_df=full_solar_df)
    price_df = pd.read_csv(optimization_configs["price_df_path"])
    solar_exp_df = pd.read_csv(optimization_configs["solar_experiment_df_path"], parse_dates=['Date']).drop(columns=['y'])
    load_exp_df = pd.read_csv(optimization_configs["load_experiment_df_path"], parse_dates=['Date']).drop(columns=['y'])
    forecast_data = ForecastData(exp_ir_df=solar_exp_df, exp_load_df=load_exp_df)
    copula_model = DailyCopulaErrorModel(pv_pipeline_path=optimization_configs["pv_copula_pipeline_path"],
                                         load_pipeline_path=optimization_configs["load_copula_pipeline_path"])

    n_points_1day = optimization_configs["n_points_1day"]
    n_horizon_days = optimization_configs.get("n_horizon_days", 7)
    resolution_in_hour = 24 / n_points_1day
    mode = optimization_configs["mode"]
    n_realizations = optimization_configs["n_realizations"]
    if not 0 <= arg.plot_scenario < n_realizations:
        raise ValueError(f"--plot_scenario must be in [0, {n_realizations - 1}]")

    # Optional soft green budget: factor * forecasted load energy over the horizon (null -> no budget)
    factor = optimization_configs.get("green_purchase_volume_factor")
    if factor is None:
        green_quota, weight_green_slack = [np.inf], None
    else:
        forecasted_PL = forecast_data.get_forecasted_load(planning_date)
        green_quota = [factor * forecasted_PL.sum() * resolution_in_hour]
        weight_green_slack = optimization_configs.get("weight_green_slack", 20)

    fig_path = os.path.join(arg.saved_path, f"planning_{planning_date}")
    os.makedirs(fig_path, exist_ok=True)
    expected_costs = []
    for n_RE_days in arg.n_RE_days:
        result_path = os.path.join(fig_path, f"plan7day_opt_{n_RE_days}REdays.csv")
        if solve_mode == 'optimize':
            one_batch_params = RE100EMS(
                battery=battery,
                measurement_data=measurement_data,
                forecast_data=forecast_data,
                planning_date=planning_date,
                n_points_1day=n_points_1day,
                n_horizon_days=n_horizon_days,
                PV_cap_original=optimization_configs["PV_cap_original"],
                PV_cap_target=optimization_configs["Pv_cap_target"],
                CFE_block_size=n_points_1day,  # one CFE block = one day
                n_CFE_blocks=n_RE_days,
                price_df=price_df,
                weight_hourly_cons=optimization_configs["weight_hourly_cons"],
                smooth_charge_factor=optimization_configs.get("smooth_charge_factor", 1),
                terminal_socs=optimization_configs["terminal_socs"],
                n_green_sources=1,
                green_allocation_quota=green_quota,
                weight_green_slack=weight_green_slack,
                copula_model=copula_model,
                n_realizations=n_realizations,
                seed=optimization_configs.get("seed", 42),
                extract_realizations=list(range(n_realizations)),  # keep every scenario's RT decisions
                mode=mode,
            )

            t0 = time.time()
            _, result_df, fval, exit_flag = one_batch_params.optimize(solver=cp.GUROBI)
            # Real-time recourse on measured data -> realized DA/RT costs (bare Pgreen_RT/Pnongreen_RT/Psell_RT columns)
            result_df = RE100EMS.adjust_RT_results_df(result_df, n_points_1hour=n_points_1day // 24, mode=mode)
            # Per-scenario RT decisions of the stochastic program: {var}_RT_{r} -> {var}_scenario_RT_{r}
            result_df = result_df.rename(columns={f"{var}_RT_{r}": f"{var}_scenario_RT_{r}"
                                                  for var in ("Pgreen", "Pnongreen", "Psell") for r in range(n_realizations)})
            result_df['expected_cost'] = fval
            result_df.to_csv(result_path, index=False)
            print("time taken (s): ", time.time() - t0)
            print(f"n_RE_days: {n_RE_days}, fval: {fval}, exit_flag: {exit_flag}")
            print(f"At planning date: {planning_date}")
            print(f"with anticipated RE100 days: {n_RE_days} days from Horizon {n_horizon_days} days")
            print(f"Expected cost over {optimization_configs['n_realizations']} scenarios = {fval:.2f} THB")
            print(f"Day-ahead green / non-green purchase (kWh): {result_df['Pgreen_DA'].sum()*resolution_in_hour:.2f} / "
                  f"{result_df['Pnongreen_DA'].sum()*resolution_in_hour:.2f}")
            print(f"Real-time green / non-green purchase on measured data (kWh): {result_df['Pgreen_RT'].sum()*resolution_in_hour:.2f} / "
                  f"{result_df['Pnongreen_RT'].sum()*resolution_in_hour:.2f}")
        elif solve_mode == 'plot':
            result_df = pd.read_csv(result_path, parse_dates=['Datetime'])
        expected_costs.append(result_df['expected_cost'].iloc[0])

        fig = plot_one_batch_one_green_operation(result_df=result_df,
                                                 figsize=optimization_configs.get("operation_fig_size", (12, 8)),
                                                 tick_fontsize=optimization_configs.get("tick_fontsize", 16),
                                                 label_fontsize=optimization_configs.get("label_fontsize", 16),
                                                 legend_fontsize=optimization_configs.get("legend_fontsize", 14),
                                                 mode=mode,
                                                 rt_scenario=arg.plot_scenario)
        fig.savefig(os.path.join(fig_path, f"plan7day_ts_{n_RE_days}REdays.pdf"), bbox_inches='tight')

    # expected cost vs anticipated RE days (bars are labelled 1..len, so only plot a full 1..N sweep)
    if list(arg.n_RE_days) == list(range(1, len(arg.n_RE_days) + 1)):
        fig = plot_REdays_net_cost(net_costs=expected_costs, figsize=optimization_configs.get("operation_fig_size", (8, 6)),
                                   tick_fontsize=optimization_configs.get("tick_fontsize", 16),
                                   label_fontsize=optimization_configs.get("label_fontsize", 16),
                                   title_fontsize=optimization_configs.get("title_fontsize", 18),
                                   planning_date=planning_date)
        fig.savefig(os.path.join(fig_path, f"plan7day_netcost_varyREdays.pdf"), bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planning_date", type=str, default="2024-09-01", help="date for planning")
    parser.add_argument("--optimization_configs_path", type=str, default="./configs/stochastic_one_batch_configs.json", help="path to optimization configs")
    parser.add_argument("--solve_mode", type=str, default="optimize", help="optimize or plot")
    parser.add_argument("--saved_path", type=str, default="./results/stochastic_one_batch_vary_RE", help="path to save results")
    parser.add_argument("--n_RE_days", type=int, nargs="+", default=list(range(1, 8)), help="anticipated RE days to solve")
    parser.add_argument("--plot_scenario", type=int, default=0, help="scenario whose real-time decisions are plotted")
    args = parser.parse_args()
    main(args)
