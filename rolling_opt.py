import os
import json
import pandas as pd
import argparse
import cvxpy as cp
import numpy as np
from src.re100_sp import BatteryParams, MeasurementData, ForecastData, RE100EMS
import shutil
from datetime import datetime
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
    n_points_1day = optimization_configs["n_points_1day"]
    PV_cap_original = optimization_configs["PV_cap_original"]
    Pv_cap_target = optimization_configs["Pv_cap_target"]
    n_horizon_days = optimization_configs.get("n_horizon_days", 7)
    
    saved_path = arg.saved_path
    os.makedirs(saved_path, exist_ok=True)
    solve_mode = arg.solve_mode  # 'optimize' or 'plot'
    continue_planning = arg.continue_planning
    if continue_planning:
        if not os.path.isdir(saved_path):
            print("No existing results folder; starting from the beginning.")
            start_planning_date = optimization_configs.get("start_planning_date", "2024-06-07")
        else:
            existing_files = [
                f for f in os.listdir(saved_path)
                if f.startswith('collected_rolling_result_up_to_') and f.endswith('.csv')
            ]
            if existing_files:
                # pick the latest by date suffix
                def extract_date(fname):
                    s = fname.split('_up_to_')[-1].replace('.csv','')
                    return pd.to_datetime(s, format='%Y-%m-%d', errors='coerce')
                latest = max(existing_files, key=lambda x: extract_date(x))
                last_planning_date = latest.split('_up_to_')[-1].replace('.csv', '')
                print(f"Continuing planning from last planning date: {last_planning_date}")
                start_planning_date = (pd.to_datetime(last_planning_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") # start from the day after the last planning date
                collected_result_df = pd.read_csv(os.path.join(saved_path, latest))
                collected_one_day_rolling_result_df = pd.read_csv(os.path.join(saved_path, f'one_day_rolling_result_up_to_{last_planning_date}.csv'))
            else:
                print("No existing results found, starting from the beginning.")
                start_planning_date = optimization_configs.get("start_planning_date", "2024-06-07")
       
    else:
        start_planning_date = optimization_configs.get("start_planning_date", "2024-06-07")
        collected_result_df = pd.DataFrame() # to collect all results
        collected_one_day_rolling_result_df = pd.DataFrame() # slice just for one-day rolling results which are the actual actions taken
    end_planning_date = optimization_configs.get("end_planning_date", "2025-05-07")
    planning_dates = pd.date_range(start=start_planning_date, end=end_planning_date, freq='D').strftime('%Y-%m-%d').tolist()
    # read measurement data
    full_solar_df = pd.read_csv(optimization_configs["solar_measurement_df_path"], parse_dates=['Datetime'])
    full_load_df = pd.read_csv(optimization_configs["load_measurement_df_path"], parse_dates=['Datetime']).rename(columns={'L':'PL'})

    measurement_data = MeasurementData(full_load_df=full_load_df, full_solar_df=full_solar_df)
    price_df = pd.read_csv(optimization_configs["price_df_path"])
    solar_exp_df = pd.read_csv(optimization_configs["solar_experiment_df_path"], parse_dates=['Date']).drop(columns=['y'])
    load_exp_df = pd.read_csv(optimization_configs["load_experiment_df_path"], parse_dates=['Date']).drop(columns=['y'])
    forecast_data = ForecastData(exp_ir_df=solar_exp_df, exp_load_df=load_exp_df)
    

    terminal_socs = optimization_configs["terminal_socs"]
    n_green_sources = optimization_configs["n_green_sources"]
    green_allocation_quota = optimization_configs["green_allocation_quota"]
    mode = optimization_configs["mode"]
    n_realizations = optimization_configs.get("n_realizations")
    extract_realizations = None
    ir_pl_error_mean = np.load(optimization_configs.get("ir_pl_error_mean_path"))
    ir_pl_error_cov = np.load(optimization_configs.get("ir_pl_error_cov_path"))

    last_saved_date = None
    
    if solve_mode == 'optimize':
        count = 0
        # Create list for store initial SOCs per each planning date
        init_socs_list = [battery.init_socs[0].copy() for _ in range(n_horizon_days)] # at planning date one, the initial SOC are from the configs with the same value for all horizon days
        start_time = datetime.now()
        for planning_date in planning_dates:
            planning_date_df = pd.DataFrame()
            planning_date_one_day_rolling_df = pd.DataFrame()
            # Vary the number of RE days from 1 to n_horizon_days
            for n_RE_days in range(1, n_horizon_days+1):
                # read battery configs again to reset initial SOCs
                with open(optimization_configs["battery_configs_path"], "r") as f:
                    battery_configs = json.load(f)
                battery_configs['init_socs'] = [init_socs_list[n_RE_days-1]]  # set initial SOCs according to the number of RE days
                battery = BatteryParams(**battery_configs)
                print(f"{mode}, Planning date: {planning_date} for {n_RE_days} RE days")
                one_batch_params = RE100EMS(
                battery=battery,
                measurement_data=measurement_data,
                forecast_data=forecast_data,
                planning_date=planning_date,
                n_points_1day=n_points_1day,
                n_horizon_days=n_horizon_days,
                n_RE_days=n_RE_days,
                PV_cap_original=PV_cap_original,
                PV_cap_target=Pv_cap_target,
                price_df=price_df,
                terminal_socs=terminal_socs,
                n_green_sources=n_green_sources,
                green_allocation_quota=green_allocation_quota,
                mode=mode,
                n_realizations=n_realizations,
                extract_realizations=extract_realizations,
                ir_pl_error_mean=ir_pl_error_mean,
                ir_pl_error_cov=ir_pl_error_cov
                )
                t0 = time.time()
                
                prob, result_df, fval, exit_flag = one_batch_params.optimize(solver=cp.GUROBI)
                next_soc = prob.var_dict['soc'].value[n_points_1day] # use the first value of the next day as the initial SOC for the next planning date
                result_df['planning_date'] = planning_date
                result_df['n_RE_days'] = n_RE_days
                # Concat result
                planning_date_df = pd.concat([planning_date_df, result_df], ignore_index=True)
                if (mode == 'prescient') and (planning_date == '2025-05-24'): # save nor just one day rolling in the last batch of prescient mode
                    planning_date_one_day_rolling_df = pd.concat([planning_date_one_day_rolling_df, result_df], ignore_index=True)
                else:
                    planning_date_one_day_rolling_df = pd.concat([planning_date_one_day_rolling_df, result_df.iloc[:n_points_1day]], ignore_index=True)
                
                print("time taken (s): ", time.time() - t0)
                print(f"n_RE_days: {n_RE_days}, fval: {fval}, exit_flag: {exit_flag}")
                print(f"Cost in planning = {fval}, exit flag = {exit_flag}")
                resolution_in_hour = 15/60  # 15 minutes resolution
                imported_green_cols = [f'Pgreen_UGT{source_idx}_DA' for source_idx in range(1, n_green_sources+1)]
                print(f"At planning date: {planning_date}")
                print(f"PV capacity (kW): {Pv_cap_target:.2f} kW")
                print(f"Battery capacity (kWh): {battery.capacity[0]:.2f} kWh")
                print(f"with anticipated RE100 days: {n_RE_days} days from Horizon {n_horizon_days} days")
                if mode != "stochastic":
                    print(f"Imported green energy in planning (kWh): {result_df[imported_green_cols].sum().sum()*resolution_in_hour:.2f} kWh")
                    for source_idx in range(1, n_green_sources+1):
                        col_name = f'Pgreen_UGT{source_idx}_DA'
                        print(f"  - from UGT{source_idx}: {result_df[col_name].sum()*resolution_in_hour:.2f} kWh")
                print(f"The next initial SOCs for battery for next planning date: {next_soc}")
                init_socs_list[n_RE_days-1] = next_soc  # update the initial SOCs for the next planning date
            count += 1
            collected_result_df = pd.concat([collected_result_df, planning_date_df], ignore_index=True)
            collected_one_day_rolling_result_df = pd.concat([collected_one_day_rolling_result_df, planning_date_one_day_rolling_df], ignore_index=True)
            # Saved result every 30 days
            if count % 30 == 0:
                os.makedirs(saved_path, exist_ok=True)
                # Delete the old ones
                if last_saved_date is not None:
                    old_file_path = os.path.join(saved_path, f'collected_rolling_result_up_to_{last_saved_date}.csv')
                    one_day_rolling_old_file_path = os.path.join(saved_path, f'one_day_rolling_result_up_to_{last_saved_date}.csv')
                    print(f"Checking for old file to delete: {old_file_path}")
                    if os.path.exists(old_file_path):
                        os.remove(old_file_path)
                        os.remove(one_day_rolling_old_file_path)
                        print(f"Deleted old file: {old_file_path}")
                        print(f"Deleted old file: {one_day_rolling_old_file_path}")
                # Save the new ones
                current_saved_date = planning_date
                collected_result_df.to_csv(os.path.join(saved_path, f'collected_rolling_result_up_to_{current_saved_date}.csv'), index=False)
                collected_one_day_rolling_result_df.to_csv(os.path.join(saved_path, f'one_day_rolling_result_up_to_{current_saved_date}.csv'), index=False)
                last_saved_date = planning_date
                print(f"last save at {datetime.now()}")

        # Save final results
        collected_result_df.to_csv(os.path.join(saved_path, f'collected_rolling_result.csv'), index=False)
        collected_one_day_rolling_result_df.to_csv(os.path.join(saved_path, f'one_day_rolling_result.csv'), index=False)
        print(f"Optimization completed and results saved at time {datetime.now()}.")
        # print in hour:minute:second format
        end_time = datetime.now()
        total_time = end_time - start_time
        total_seconds = int(total_time.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Total optimization time: {hours}:{minutes:02}:{seconds:02}")
    elif solve_mode == 'plot':
        pass

    
            
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization_configs_path", type=str, default="./configs/prescient_rolling_configs.json", help="path to optimization configs")
    parser.add_argument("--continue_planning", type=bool, default=False, help="whether to continue planning from existing results")
    parser.add_argument("--solve_mode", type=str, default="optimize", help="optimize or plot")
    parser.add_argument("--saved_path", type=str, default="./results/prescient_rolling", help="path to save results")
    args = parser.parse_args()
    main(args)