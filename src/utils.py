import pandas as pd
import datetime
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
        