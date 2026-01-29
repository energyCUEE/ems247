import os
import json
import pandas as pd
import argparse
import cvxpy as cp
import numpy as np
from src.re100_sp import BatteryParams, MeasurementData, ForecastData, RE100EMS
from src.plots import plot_one_batch_one_green_operation, plot_REdays_net_cost
from src.utils import calculate_cost
import warnings
warnings.filterwarnings("ignore")
import time
def main(arg):
    with open(arg.optimization_configs_path, "r") as f:
        optimization_configs = json.load(f)
    # battery configs
    with open(optimization_configs["battery_configs_path"], "r") as f:
        battery_configs = json.load(f)

    battery = BatteryParams(**battery_configs)
    planning_date = arg.planning_date
    saved_path = arg.saved_path
    solve_mode = arg.solve_mode  # 'optimize' or 'plot'

    # read measurement data
    full_solar_df = pd.read_csv(optimization_configs["solar_measurement_df_path"], parse_dates=['Datetime'])
    full_load_df = pd.read_csv(optimization_configs["load_measurement_df_path"], parse_dates=['Datetime']).rename(columns={'L':'PL'})

    measurement_data = MeasurementData(full_load_df=full_load_df, full_solar_df=full_solar_df)
    price_df = pd.read_csv(optimization_configs["price_df_path"])
    solar_exp_df = pd.read_csv(optimization_configs["solar_experiment_df_path"], parse_dates=['Date']).drop(columns=['y'])
    load_exp_df = pd.read_csv(optimization_configs["load_experiment_df_path"], parse_dates=['Date']).drop(columns=['y'])
    forecast_data = ForecastData(exp_ir_df=solar_exp_df, exp_load_df=load_exp_df)
    
    n_points_1day = optimization_configs["n_points_1day"]
    PV_cap_original = optimization_configs["PV_cap_original"]
    Pv_cap_target = optimization_configs["Pv_cap_target"]
    n_horizon_days = optimization_configs.get("n_horizon_days", 7)
    smooth_charge_factor = optimization_configs.get("smooth_charge_factor", 1/3)
    # n_horizon_days = 7  # fixed at 7 days horizon
    terminal_socs = optimization_configs["terminal_socs"]
    weight_hourly_cons = optimization_configs["weight_hourly_cons"]  
    n_green_sources = optimization_configs["n_green_sources"]
    green_allocation_quota = optimization_configs["green_allocation_quota"]
    mode = optimization_configs["mode"]
    fig_path = os.path.join(saved_path, f"planning_{planning_date}")
    os.makedirs(fig_path, exist_ok=True)
    net_costs = []
    for n_RE_days in range(1, n_horizon_days+1):
        if solve_mode == 'optimize':
            one_batch_params = RE100EMS(
            battery=battery,
            measurement_data=measurement_data,
            forecast_data=forecast_data,
            planning_date=planning_date,
            n_points_1day=n_points_1day,
            n_horizon_days=n_horizon_days,
            PV_cap_original=PV_cap_original,
            PV_cap_target=Pv_cap_target,
            n_RE_days=n_RE_days,
            price_df=price_df,
            weight_hourly_cons=weight_hourly_cons,
            smooth_charge_factor=smooth_charge_factor,
            terminal_socs=terminal_socs,
            n_green_sources=n_green_sources,
            green_allocation_quota=green_allocation_quota,
            mode=mode,
            )


            t0 = time.time()
            
            _, result_df, fval, exit_flag = one_batch_params.optimize(solver=cp.GUROBI)
            result_df.to_csv(os.path.join(fig_path, f"plan7day_opt_{n_RE_days}REdays.csv"), index=False)
            print("time taken (s): ", time.time() - t0)
            net_cost = calculate_cost(result_df=result_df, mode=mode, is_rolling=False, n_green_sources=n_green_sources)
            net_costs.append(net_cost)
            print(f"n_RE_days: {n_RE_days}, fval: {fval}, exit_flag: {exit_flag}")
            print(f"Cost in planning = {fval}, exit flag = {exit_flag}")
            resolution_in_hour = 15/60  # 15 minutes resolution
            imported_green_cols = [f'Pgreen_UGT{idx+1}_DA' for idx in range(n_green_sources)]
            print(f"At planning date: {planning_date}")
            print(f"PV capacity (kW): {Pv_cap_target:.2f} kW")
            print(f"Battery capacity (kWh): {battery.capacity[0]:.2f} kWh")
            print(f"with anticipated RE100 days: {n_RE_days} days from Horizon {n_horizon_days} days")
            print(f"Imported green energy in planning (kWh): {result_df[imported_green_cols].sum().sum()*resolution_in_hour:.2f} kWh")
            for source_idx in range(1, n_green_sources+1):
                col_name = f'Pgreen_UGT{source_idx}_DA'
                print(f"  - from UGT{source_idx}: {result_df[col_name].sum()*resolution_in_hour:.2f} kWh")
        elif solve_mode == 'plot':
            result_df = pd.read_csv(os.path.join(fig_path, f"plan7day_opt_{n_RE_days}REdays.csv"), parse_dates=['Datetime'])
            net_cost = calculate_cost(result_df=result_df, mode=mode, is_rolling=False, n_green_sources=n_green_sources)
            net_costs.append(net_cost)
        
        figsize = optimization_configs.get("operation_fig_size", (12, 8))
        tick_fontsize = optimization_configs.get("tick_fontsize",16)
        label_fontsize = optimization_configs.get("label_fontsize",16)
        legend_fontsize = optimization_configs.get("legend_fontsize",14)
        title_fontsize = optimization_configs.get("title_fontsize",18)
        fig = plot_one_batch_one_green_operation(result_df=result_df, figsize=figsize,
                                                  tick_fontsize=tick_fontsize,
                                                  label_fontsize=label_fontsize,
                                                  legend_fontsize=legend_fontsize,
                                                  mode=mode,)
        # Save the figure as pdf
        fig.savefig(os.path.join(fig_path, f"plan7day_ts_{n_RE_days}REdays.pdf"), bbox_inches='tight')
        
    # save net costs
    fig = plot_REdays_net_cost(net_costs=net_costs, figsize=optimization_configs.get("operation_fig_size", (8, 6)),
                               tick_fontsize=optimization_configs.get("tick_fontsize",16),
                               label_fontsize=optimization_configs.get("label_fontsize",16),
                               title_fontsize=optimization_configs.get("title_fontsize",18))
    fig.savefig(os.path.join(fig_path, f"plan7day_netcost_varyREdays.pdf"), bbox_inches='tight')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planning_date", type=str, default="2024-09-01", help="date for planning")
    parser.add_argument("--optimization_configs_path", type=str, default="./configs/deterministic_one_batch_configs.json", help="path to optimization configs")
    parser.add_argument("--solve_mode", type=str, default="optimize", help="optimize or plot")
    parser.add_argument("--saved_path", type=str, default="./results/deterministic_one_batch_vary_RE", help="path to save results")
    args = parser.parse_args()
    main(args)