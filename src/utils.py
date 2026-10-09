import pandas as pd
import datetime
import numpy as np
def align_to_forecast_start(measurement_df, dt_col='Datetime',target_col='PL', forecast_steps=96*7, resolution='15min'):
    # Align the measurement DataFrame to the forecast start
    _df = measurement_df[[dt_col, target_col]].copy()
    for i in range(1, forecast_steps + 1):
        _df[f'ynext{i}'] = _df[target_col].shift(-i)

    # extract only step before midnight 00.00
    _df['time'] = _df[dt_col].dt.time
    _df['Date'] = _df[dt_col].dt.date
    _df['Date'] = pd.to_datetime(_df['Date'])
    ## forecast start time is 1 step before midnight
        # midnight as Timestamp
    midnight = pd.to_datetime('00:00:00')

    # subtract resolution
    forecast_start_time = (midnight - pd.to_timedelta(resolution)).time()
    aligned_df = _df[_df['time'] == forecast_start_time].drop(columns=['time', dt_col, target_col]).reset_index(drop=True)
    aligned_df = aligned_df.dropna().reset_index(drop=True)
    return aligned_df

def create_error_matrix(aligned_measurement_df, train_exp_df, forecast_steps=96*7):

    # ensure that both dataframes have 'Date' that is datetime type col
    assert pd.api.types.is_datetime64_any_dtype(aligned_measurement_df['Date']), "aligned_measurement_df['Date'] is not datetime"
    assert pd.api.types.is_datetime64_any_dtype(train_exp_df['Date']), "train_exp_df['Date'] is not datetime"

    
    _sliced_meas_df = aligned_measurement_df[aligned_measurement_df['Date'].isin(train_exp_df['Date'])].copy()

    ynext_cols = [f"ynext{step}" for step in range(1, forecast_steps + 1)]

    ytrue_arr = _sliced_meas_df[ynext_cols].clip(lower=0).to_numpy()
    ypred_arr = train_exp_df.sort_values(by='Date')[ynext_cols].clip(lower=0).to_numpy()

    # To be added to the deterministic forecast, use the definition of error = ytrue - ypred
    error_matrix = ytrue_arr - ypred_arr

    return error_matrix

def create_error_matrix_pv(
                           train_exp_df, 
                           full_solar_df,
                           forecast_steps=96*7,
                           pv_cap_original=3.75,
                           pv_cap_target=15):
    predicted_pv_list = []
    actual_pv_list = []
    for date in train_exp_df['Date']:
        if date.date() not in full_solar_df['Datetime'].dt.date.unique():
            raise ValueError(f"Date {date.date()} in train_exp_df is not present in aligned_measurement_df.")
        # for date 2025-01-01, the target range is 2025-01-02 00:00:00 to 2025-02-08 23:45:00, which is 7 days after the date
        target_time_range = pd.date_range(start=pd.Timestamp(date) + pd.Timedelta(days=1), periods=forecast_steps, freq='15min')
        predicted_ir = train_exp_df[train_exp_df['Date'] == date].loc[:, [f"ynext{step}" for step in range(1, forecast_steps + 1)]].to_numpy().flatten()
        # pull scaling factor from full_solar_df, for the date, get the PV value at 00:00:00, which is the first row of the day
        scaling_factor = full_solar_df[full_solar_df['Datetime'].isin(target_time_range)]['daily_alpha'].to_numpy().flatten()
        actual_pv = full_solar_df[full_solar_df['Datetime'].isin(target_time_range)]['adjusted_PV'].to_numpy().flatten()
        # check if predicted_ir and scaling_factor have the same length
        if len(predicted_ir) != len(scaling_factor):
            raise ValueError(f"Length mismatch for date {date}: predicted_ir has length {len(predicted_ir)}, scaling_factor has length {len(scaling_factor)}.")
        # scale the predicted_ir by the scaling_factor
        predicted_pv = predicted_ir * scaling_factor
        predicted_pv_list.append(predicted_pv)
        actual_pv_list.append(actual_pv)
    predicted_pv_arr = np.array(predicted_pv_list)
    actual_pv_arr = np.array(actual_pv_list)
    # scale the predicted and actual PV by the ratio of target capacity to original capacity
    scaling_ratio = pv_cap_target / pv_cap_original
    scaled_predicted_pv = predicted_pv_arr * scaling_ratio
    scaled_actual_pv = actual_pv_arr * scaling_ratio    
    error_matrix = scaled_actual_pv - scaled_predicted_pv
    return error_matrix, scaled_actual_pv, scaled_predicted_pv
    

def calculate_cost(result_df, mode='deterministic', is_rolling=False, n_green_sources=3, 
                   resolution_in_hour=15/60, case='sell'):
    
    imported_Pgreen_cols = [f'Pgreen_UGT{idx+1}_DA' for idx in range(n_green_sources)]
    imported_green_planning_rates = [f'UGT{idx+1}_PLANNING_BUY_RATE' for idx in range(n_green_sources)]
    if not is_rolling:
        green_buy_cost = sum(
                            result_df[imported_Pgreen_cols[idx]] * result_df[imported_green_planning_rates[idx]]
                            for idx in range(n_green_sources)
                        ).sum() * resolution_in_hour
        nongreen_buy_cost = (result_df['posPnet_DA'] * result_df['TOU_PLANNING_SELL_RATE']).sum() * resolution_in_hour
        if case=='sell':
            sell_cost = (result_df['negPnet_DA'] * result_df['TOU_PLANNING_SELL_RATE']).sum() * resolution_in_hour
    
        
            net_cost = green_buy_cost + nongreen_buy_cost - sell_cost
        else:
            net_cost = green_buy_cost + nongreen_buy_cost
    return net_cost
        