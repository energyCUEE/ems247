from re import I
import re
from tabnanny import verbose
import pandas as pd
import numpy as np
import numpy.matlib
import os
import datetime
import math
import cvxpy as cp
from typing import List, Tuple, Optional

from pydantic import constr
from regex import F

class BatteryParams:
    """
    Class representing a set of batteries with various operational limits and efficiencies.
    
    Attributes:
        _no_batts (int): Number of batteries.
        _init_socs (np.ndarray): Initial state of charge for each battery.
        _min_socs (np.ndarray): Minimum state of charge for each battery.
        _max_socs (np.ndarray): Maximum state of charge for each battery.
        _capacity (np.ndarray): Capacity of each battery.
        _max_charge_rates (np.ndarray): Maximum charging rate for each battery.
        _max_discharge_rates (np.ndarray): Maximum discharging rate for each battery.
        _charge_effs (np.ndarray): Charging efficiency for each battery.
        _discharge_effs (np.ndarray): Discharging efficiency for each battery.
    """
    def __init__(self, no_batts: int, 
                 init_socs: List[float],
                 min_socs: List[float], 
                 max_socs: List[float],
                 capacity: List[float], 
                 max_charge_rates: List[float], max_discharge_rates: List[float], 
                 charge_effs: List[float], discharge_effs: List[float],
                 ) -> None:
        """
        Initializes the Battery class by validating input lists and converting them to NumPy arrays.
        
        Args:
            no_batts (int): Number of batteries.
            init_socs (List[float]): Initial state of charges.
            min_socs (List[float]): Minimum state of charges.
            max_socs (List[float]): Maximum state of charges.
            capacity (List[float]): Capacities of the batteries.
            max_charge_rates (List[float]): Maximum charging rates.
            max_discharge_rates (List[float]): Maximum discharging rates.
            charge_effs (List[float]): Charging efficiencies.
            discharge_effs (List[float]): Discharging efficiencies.
        
        Raises:
            ValueError: If no_batts is not a positive integer or any list is not of proper length or type.
        """
        if not isinstance(no_batts, int) or no_batts <= 0:
            raise ValueError("no_batts must be a positive integer.")

        # Dictionary of attributes to validate and convert
        attributes = {
            "init_socs": init_socs,
            "min_socs": min_socs,
            "max_socs": max_socs,
            "capacity": capacity,
            "max_charge_rates": max_charge_rates,
            "max_discharge_rates": max_discharge_rates,
            "charge_effs": charge_effs,
            "discharge_effs": discharge_effs,
        }

        # Validate each attribute and convert to a NumPy array with shape (-1,)
        for attr_name, attr_value in attributes.items():
            if not isinstance(attr_value, list):
                raise ValueError(f"{attr_name} must be provided as a list.")
            if len(attr_value) != no_batts:
                raise ValueError(f"{attr_name} must have length equal to no_batts.")
            setattr(self, f"{attr_name}", np.array(attr_value).reshape(-1,))
        
        self.no_batts: int = no_batts

class MeasurementData:
    def __init__(self, 
                 full_solar_df: pd.DataFrame,
                 full_load_df: pd.DataFrame,):

        # raise error if solar_df does not contain a 'Datatime' and 'PV' column
        if 'Datetime' not in full_solar_df.columns or 'adjusted_PV' not in full_solar_df.columns:
            raise ValueError("full_solar_df must contain 'Datetime' and 'PV' columns.")
        
        # raise error if load_df does not contain a 'Datetime' and 'PL' column
        if 'Datetime' not in full_load_df.columns or 'PL' not in full_load_df.columns:
            raise ValueError("full_load_df must contain 'Datetime' and 'PL' columns.")
        self._full_solar_df = full_solar_df.set_index('Datetime')
        self._full_load_df = full_load_df.set_index('Datetime')

    def get_PV_data(self, start_time: str, end_time: str, PV_cap_original: float, PV_cap_target: float) -> pd.Series:
        """
        Get solar PV data between start_time and end_time.
        
        Args:
            start_time (str): Start time in 'YYYY-MM-DD HH:MM:SS' format.
            end_time (str): End time in 'YYYY-MM-DD HH:MM:SS' format.
        
        Returns:
            1d np.ndarray: Solar PV data between the specified times.
        """
        return self._full_solar_df.loc[start_time:end_time, 'adjusted_PV'].clip(lower=0).to_numpy() * PV_cap_target / PV_cap_original
    
    def get_PL_data(self, start_time: str, end_time: str) -> pd.Series:
        """
        Get load PL data between start_time and end_time.
        
        Args:
            start_time (str): Start time in 'YYYY-MM-DD HH:MM:SS' format.
            end_time (str): End time in 'YYYY-MM-DD HH:MM:SS' format.
        
        Returns:
            1d np.ndarray: Load PL data between the specified times.
        """
        return self._full_load_df.loc[start_time:end_time, 'PL'].clip(lower=0).to_numpy()
    
    def get_alpha(self, start_time: str, end_time:str):
        
        return self._full_solar_df.loc[start_time:end_time, 'daily_alpha'].to_numpy()

class ForecastData:
    def __init__(self,
                 exp_ir_df: pd.DataFrame,
                 exp_load_df: pd.DataFrame,
                 index_col: str = 'Date') -> None:
        
        self._exp_ir_df = exp_ir_df.set_index(index_col)
        self._exp_load_df = exp_load_df.set_index(index_col)

    def get_forecasted_ir(self, planning_date: str) -> pd.Series:
        """
        Get forecasted irradiance data for the specified planning time.
        
        Args:
            planning_date (str): Planning time in any format of original
        """
        return self._exp_ir_df.loc[planning_date].clip(lower=0).to_numpy()

    def get_forecasted_load(self, planning_date: str) -> pd.Series:
        """
        Get forecasted load data for the specified planning time.

        Args:
            planning_date (str): Planning time in any format of original
        """
        return self._exp_load_df.loc[planning_date].clip(lower=0).to_numpy()

    def get_ir_realizations(self, planning_date: str, ir_error_mean: np.ndarray,ir_error_cov: np.ndarray, n_realizations: int, seed: int=42) -> pd.Series:
        """
        Get realized irradiance data for the specified planning time separately.

        Args:
            planning_date (str): Planning time in any format of original
            ir_error_mean (np.ndarray): Error mean vector for irradiance with dimension (forecast_steps,). 1D array.
            ir_error_cov (np.ndarray): Error covariance matrix for irradiance with dimension (forecast_steps, forecast_steps).

        Returns:
            np.ndarray: realizations of irradiance data with shape (n_realizations, forecast_steps).
        """
        # Get the deterministic forecasted irradiance as mean
        forecasted_ir = self.get_forecasted_ir(planning_date)
        
        # Get the error realizations
        np.random.seed(seed)
        error_realizations = np.random.multivariate_normal(ir_error_mean, ir_error_cov, size=n_realizations)
        
        # Add the error realizations to the deterministic forecast
        ir_realizations = forecasted_ir[None, :] + error_realizations

        return ir_realizations.clip(min=0)

    def get_pl_realizations(self, planning_date: str, pl_error_mean: np.ndarray,pl_error_cov: np.ndarray, n_realizations: int, seed: int=42) -> pd.Series:
        """
        Get realized load data for the specified planning time separately.

        Args:
            planning_date (str): Planning time in 'YYYY-MM-DD HH:MM:SS' format.
            pl_error_mean (np.ndarray): Error mean vector for load with dimension (forecast_steps,). 1D array.
            pl_error_cov (np.ndarray): Error covariance matrix for load with dimension (forecast_steps, forecast_steps).

        Returns:
            np.ndarray: realizations of load data with shape (n_realizations, forecast_steps).
        """
        # Get the deterministic forecasted load as mean
        forecasted_pl = self.get_forecasted_load(planning_date)
        
        # Get the error realizations
        np.random.seed(seed)
        error_realizations = np.random.multivariate_normal(pl_error_mean, pl_error_cov, size=n_realizations)
 
        # Add the error realizations to the deterministic forecast
        pl_realizations = forecasted_pl[None, :] + error_realizations
    
        return pl_realizations.clip(min=0)

    def get_ir_pl_realizations(self, planning_date: str, ir_pl_error_cov: np.ndarray, ir_pl_error_mean: np.ndarray, n_realizations: int, seed: int=42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get realized irradiance and load data for the specified planning time together from stack

        Args:
            planning_date (str): Planning time in 'YYYY-MM-DD HH:MM:SS' format.
            ir_pl_error_cov (np.ndarray): Error covariance matrix for irradiance and load with dimension (2*forecast_steps, 2*forecast_steps).
            ir_pl_error_mean (np.ndarray): Error mean vector for irradiance and load with dimension (2*forecast_steps,). 1D array.
            n_realizations (int): Number of realizations to generate.

        Returns:
            Tuple[np.ndarray, np.ndarray]: realizations of irradiance and load data with shape (n_realizations, forecast_steps).
        """
        # Get the deterministic forecasted irradiance and load as mean
        forecasted_ir = self.get_forecasted_ir(planning_date)
        forecasted_pl = self.get_forecasted_load(planning_date)
        # assert the dimension of deterministic forecasted data
        assert len(forecasted_ir) == len(forecasted_pl), "The dimension of forecasted_ir and forecasted_pl must be the same."
        # Get the error realizations
        np.random.seed(seed)
        error_realizations = np.random.multivariate_normal(
            ir_pl_error_mean, ir_pl_error_cov, size=n_realizations)

        ir_realizations = forecasted_ir[None, :] + error_realizations[:, :len(forecasted_ir)]
        pl_realizations = forecasted_pl[None, :] + error_realizations[:, len(forecasted_ir):]

        return ir_realizations.clip(min=0), pl_realizations.clip(min=0)
    
    
    

class RE100EMS:

    def __init__(self,
                 battery: BatteryParams,
                 measurement_data: MeasurementData,
                 forecast_data: ForecastData=None,
                 price_df: pd.DataFrame=None,
                 planning_date: str='2025-01-01',
                 n_points_1day:int=96,
                 n_horizon_days: int=7,
                 PV_cap_original: float=4,
                 PV_cap_target: float=30,
                 n_RE_days: int=4,
                 smooth_charge_factor: float=1,
                  terminal_socs: List[float]=None,
                  weight_hourly_cons: int=1,
                  weight_charge_batt: int=None,
                    weight_smooth_charge: int=None,
                 n_realizations: int=None,
                 n_green_sources: int=1,
                 seed: int=42,
                 green_allocation_quota: List[float]=[np.inf], # kWh per month
                 ir_error_cov: np.ndarray=None,
                pl_error_cov: np.ndarray=None,
                ir_error_mean: np.ndarray=None,
                    pl_error_mean: np.ndarray=None,
                    ir_pl_error_cov: np.ndarray=None,
                    ir_pl_error_mean: np.ndarray=None,
                    extract_realizations: list=None, # [1, 23] some specific scenario indices to be extracted
                 mode: str='deterministic'):
        
        self.battery = battery
        self.n_green_sources = n_green_sources
        if price_df is None:
            self.price_df = pd.read_csv("../configs/price.csv")
        else:
            self.price_df = price_df
        
        # Raise error if the length of green_allocation_quota does not match n_green_sources
        if len(green_allocation_quota) != n_green_sources:
            raise ValueError("The length of green_allocation_quota must match n_green_sources.")
        self.green_allocation_quota = np.array(green_allocation_quota)
        
        self.planning_date = datetime.datetime.strptime(planning_date, '%Y-%m-%d')
        self.n_points_1day = n_points_1day
        self.n_horizon_days = n_horizon_days
        
        # target dates of planning
        self.start_date = self.planning_date + datetime.timedelta(days=1)
        self.end_date = self.planning_date+ datetime.timedelta(days=n_horizon_days)
        
        self.start_time = self.start_date.strftime('%Y-%m-%d 00:00:00')
        self.end_time = self.end_date.strftime('%Y-%m-%d 23:59:59')
        
        self.target_dt = pd.date_range(start=self.start_time, end=self.end_time, freq=f'{1440//n_points_1day}min')
        
        self.alpha = measurement_data.get_alpha(self.start_time, self.end_time)
        
        # get measurement data
        self.meas_PV = measurement_data.get_PV_data(self.start_time, self.end_time, PV_cap_original, PV_cap_target)
        self.meas_PL = measurement_data.get_PL_data(self.start_time, self.end_time)
        
       
        if forecast_data is None:
            raise ValueError("forecast_data must be provided for deterministic or stochastic mode.")
        # get forecast data
        self.forecasted_ir = forecast_data.get_forecasted_ir(self.planning_date.strftime('%Y-%m-%d'))
        self.forecasted_PV = self.alpha * self.forecasted_ir / PV_cap_original * PV_cap_target
        
        self.forecasted_PL = forecast_data.get_forecasted_load(self.planning_date.strftime('%Y-%m-%d'))
        if mode == 'stochastic':
            if n_realizations is None or n_realizations <= 0:
                raise ValueError("n_realizations must be a positive integer for stochastic mode.")
            self.n_realizations = n_realizations

            if ir_pl_error_cov is not None and ir_pl_error_mean is not None:
                # assert the dimension of ir_pl_error_cov and ir_pl_error_mean
                assert ir_pl_error_cov.shape[0] == ir_pl_error_cov.shape[1], "The ir_pl_error_cov must be a square matrix."
                assert ir_pl_error_cov.shape[0] == len(self.forecasted_ir)*2, "The dimension of ir_pl_error_cov must be 2 times the length of forecasted_ir."
                assert len(ir_pl_error_mean) == len(self.forecasted_ir)*2, "The length of ir_pl_error_mean must be 2 times the length of forecasted_ir."
                self.ir_pl_error_cov = ir_pl_error_cov
                self.ir_pl_error_mean = ir_pl_error_mean
                self.ir_realizations, self.pl_realizations = forecast_data.get_ir_pl_realizations(
                    planning_date=self.planning_date.strftime('%Y-%m-%d'),
                    ir_pl_error_cov=self.ir_pl_error_cov,
                    ir_pl_error_mean=self.ir_pl_error_mean,
                    n_realizations=self.n_realizations,
                    seed=seed
                    )
                
                self.pv_realizations =  self.alpha[None, :] * self.ir_realizations/ PV_cap_original * PV_cap_target
                print("Using stacked ir and pl error covariance matrix to generate scenarios.")
            elif ir_error_cov is not None and ir_error_mean is not None and pl_error_cov is not None and pl_error_mean is not None:
                # assert the dimension of ir_error_cov and ir_error_mean
                assert ir_error_cov.shape[0] == ir_error_cov.shape[1], "The ir_error_cov must be a square matrix."
                assert ir_error_cov.shape[0] == len(self.forecasted_ir), "The dimension of ir_error_cov must be the length of forecasted_ir."
                assert len(ir_error_mean) == len(self.forecasted_ir), "The length of ir_error_mean must be the length of forecasted_ir."
                self.ir_error_cov = ir_error_cov
                self.ir_error_mean = ir_error_mean

                # assert the dimension of pl_error_cov and pl_error_mean
                assert pl_error_cov.shape[0] == pl_error_cov.shape[1], "The pl_error_cov must be a square matrix."
                assert pl_error_cov.shape[0] == len(self.forecasted_PL), "The dimension of pl_error_cov must be the length of forecasted_PL."
                assert len(pl_error_mean) == len(self.forecasted_PL), "The length of pl_error_mean must be the length of forecasted_PL."
                self.pl_error_cov = pl_error_cov
                self.pl_error_mean = pl_error_mean

                self.ir_realizations = forecast_data.get_ir_realizations(
                    planning_date=self.planning_date.strftime('%Y-%m-%d'),
                    ir_error_cov=self.ir_error_cov,
                    ir_error_mean=self.ir_error_mean,
                    n_realizations=self.n_realizations,
                    seed=seed
                    )
                self.pv_realizations = self.alpha[None, :] * self.ir_realizations / PV_cap_original * PV_cap_target
                
                self.pl_realizations = forecast_data.get_pl_realizations(
                    planning_date=self.planning_date.strftime('%Y-%m-%d'),
                    pl_error_cov=self.pl_error_cov,
                    pl_error_mean=self.pl_error_mean,
                    n_realizations=self.n_realizations,
                    seed=seed)
            else:
                raise ValueError("Either ir_pl_error_cov and ir_pl_error_mean or ir_error_cov, ir_error_mean, pl_error_cov, pl_error_mean must be provided for stochastic mode.")
                

        # Get buy and sell rates based on TOU tariffs
        self.TOU_PLANNING_BUY_RATE, self.TOU_PLANNING_SELL_RATE = self._get_buy_sell_rate_planning(rate_choice='THflat')
        
        # Get buy and sell rates for each green source based on UGT tariffs
        self.UGT_PLANNING_BUY_RATE :dict = {}
        self.UGT_PLANNING_SELL_RATE :dict = {}
        for source_idx in range(1, self.n_green_sources+1):
            self.UGT_PLANNING_BUY_RATE[f'UGT{source_idx}'], self.UGT_PLANNING_SELL_RATE[f'UGT{source_idx}'] = self._get_buy_sell_rate_planning(rate_choice=f'UGT{source_idx}')

        
        self.TOU_RT_BUY_RATE, self.TOU_RT_SELL_RATE = self._get_buy_sell_rate_realtime(rate_choice='THflat')
        self.UGT_RT_BUY_RATE :dict = {}
        self.UGT_RT_SELL_RATE :dict = {}
        for source_idx in range(1, self.n_green_sources+1):
            self.UGT_RT_BUY_RATE[f'UGT{source_idx}'], self.UGT_RT_SELL_RATE[f'UGT{source_idx}'] = self._get_buy_sell_rate_realtime(rate_choice=f'UGT{source_idx}')

        
        self.weight_hourly_cons: Optional[int] = weight_hourly_cons
        self.weight_charge_batt: Optional[int] = weight_charge_batt  
        self.weight_smooth_charge: Optional[int] = weight_smooth_charge
        self.terminal_socs: Optional[np.ndarray] = terminal_socs
        self.n_RE_days: Optional[int] = n_RE_days
        self.extract_realizations: Optional[list] = extract_realizations

        self.smooth_charge_factor = smooth_charge_factor
        self.mode = mode
    
    def _get_buy_sell_rate_planning(self, rate_choice) -> Tuple[np.ndarray, np.ndarray]:
        rate_df = pd.DataFrame({'Datetime': self.target_dt})
        
        # Lookup the corresponding row
        if rate_choice == 'THflat':
            row = self.price_df[self.price_df['mode'] == 'nongreen']
        else:
            row = self.price_df[self.price_df['mode'] == rate_choice]

        if row.empty:
            raise ValueError(f"Rate choice '{rate_choice}' not found in price_df")

        buy_rate = float(row['buy_dayahead'])
        sell_rate = float(row['sell_dayahead'])

        rate_df['buy'] = buy_rate
        rate_df['sell'] = sell_rate

        return rate_df['buy'].to_numpy(), rate_df['sell'].to_numpy()
    
    def _get_buy_sell_rate_realtime(self, rate_choice) -> Tuple[np.ndarray, np.ndarray]:
        rate_df = pd.DataFrame({'Datetime': self.target_dt})
        
        if rate_choice == 'THflat':
            row = self.price_df[self.price_df['mode'] == 'nongreen']
        else:
            row = self.price_df[self.price_df['mode'] == rate_choice]

        if row.empty:
            raise ValueError(f"Rate choice '{rate_choice}' not found in price_df")

        buy_rate = float(row['buy_realtime'])
        sell_rate = float(row['sell_realtime'])

        rate_df['buy'] = buy_rate
        rate_df['sell'] = sell_rate

        return rate_df['buy'].to_numpy(), rate_df['sell'].to_numpy()

    
    def set_new_mode(self, mode: str):
        """
        Set a new optimization mode for the EMS.
        
        Args:
            mode (str): The new mode to set ('deterministic' or 'stochastic').
        """
        if mode not in ['deterministic', 'stochastic']:
            raise ValueError("Mode must be either 'deterministic' or 'stochastic'.")
        self.mode = mode
        

    def _deterministic_optimization_multi_providers(self, solver=cp.GUROBI):
        length_optimvar = len(self.target_dt)
        
        resolution_in_hour = (self.target_dt[1] - self.target_dt[0]).seconds / 3600
        n_horizon_days = self.n_horizon_days
        n_horizon_hours = math.ceil((self.target_dt[-1]- self.target_dt[0]).total_seconds() / 3600)
        
        n_points_1hour = int(1 / resolution_in_hour)
        n_hourly_blocks = int(n_horizon_hours/self.weight_hourly_cons)
        n_green_sources = self.n_green_sources
        
        # Define decision variables
        Pnet = cp.Variable(length_optimvar, name='Pnet')
        posPnet = cp.Variable(length_optimvar, nonneg=True, name='posPnet')

        # negPnet_hourly = cp.Variable(n_hourly_blocks, nonneg=True, name='negPnet_hourly')
        negPnet = cp.Variable(length_optimvar, nonneg=True, name='negPnet')


        # Create only hourly decision variables and replicate them for each time step within the hour to reduce the number of decision variables
        P_hourly_green = cp.Variable((n_hourly_blocks, n_green_sources), nonneg=True, name='P_hourly_green')
        Pgreen = cp.Variable((length_optimvar, n_green_sources), nonneg=True, name='Pgreen')
        P_hourly_nongreen = cp.Variable(n_hourly_blocks, nonneg=True, name='P_hourly_nongreen')
        Pnongreen = cp.Variable(length_optimvar, nonneg=True, name='Pnongreen')
        
        # Battery
        no_batts = self.battery.no_batts
        Pdchg = cp.Variable((length_optimvar, no_batts), nonneg=True, name='Pdchg')
        Pchg  = cp.Variable((length_optimvar, no_batts), nonneg=True, name='Pchg')
        soc = cp.Variable((length_optimvar + 1, no_batts), name='soc')
        x_chg   = cp.Variable((length_optimvar, no_batts), integer=True, name='x_chg')
        x_dchg    = cp.Variable((length_optimvar, no_batts), integer=True, name='x_dchg')
        
        # Integer decision variables for grid status (using Big-M formulation later)
        u_normXnongreen1day = cp.Variable(n_horizon_days, integer=True, name='normXnongreen1Day')
        u_hourly_green_sources = cp.Variable(n_hourly_blocks, integer=True, name='u_hourly_green_sources')
        u_green_sources = cp.Variable(length_optimvar, integer=True, name='u_green_sources')
        x_hourly_green = cp.Variable((n_hourly_blocks, n_green_sources), integer=True, name='x_hourly_green')
        x_green = cp.Variable((length_optimvar, n_green_sources), integer=True, name='x_green')
        x_hourly_nongreen = cp.Variable(n_hourly_blocks, integer=True, name='x_hourly_nongreen')
        x_nongreen = cp.Variable(length_optimvar, integer=True, name='x_nongreen')
        
        # Get the buy sell rates in array format for optimization of shape (length_optimvar, n_green_sources)
        UGT_PLANNING_BUY_RATE_array = np.zeros((length_optimvar, n_green_sources))
        UGT_PLANNING_SELL_RATE_array = np.zeros((length_optimvar, n_green_sources))
        UGT_RT_BUY_RATE_array = np.zeros((length_optimvar, n_green_sources))
        UGT_RT_SELL_RATE_array = np.zeros((length_optimvar, n_green_sources))
        for source_idx in range(n_green_sources):
            UGT_PLANNING_BUY_RATE_array[:, source_idx] = self.UGT_PLANNING_BUY_RATE[f'UGT{source_idx+1}']
            UGT_PLANNING_SELL_RATE_array[:, source_idx] = self.UGT_PLANNING_SELL_RATE[f'UGT{source_idx+1}']
            UGT_RT_BUY_RATE_array[:, source_idx] = self.UGT_RT_BUY_RATE[f'UGT{source_idx+1}']
            UGT_RT_SELL_RATE_array[:, source_idx] = self.UGT_RT_SELL_RATE[f'UGT{source_idx+1}']
        print()
        # Define objective: cost of purchasing nongreen and green power, use UGT which has the highest sell rate in selling price, assume it is located at the last column
        total_obj = (self.TOU_PLANNING_BUY_RATE @ Pnongreen + cp.sum(cp.multiply(UGT_PLANNING_BUY_RATE_array, Pgreen)) 
                     )* resolution_in_hour # no selling considered in planning stage
        
        # Enable smooth charge/discharge by adding a small penalty to the objective
        if self.weight_smooth_charge is not None:
            total_obj += self.weight_smooth_charge*cp.sum(Pdchg[1:, :] - Pdchg[:-1, :])/self.battery.max_discharge_rates
        
        M: float = 500  # Big-M constant for constraints
        
        # Initialize constraints list
        constraints = []
        # Integer variable constraints (ensuring binary values 0 or 1)
        constraints.append(u_normXnongreen1day >= 0)
        constraints.append(u_normXnongreen1day <= 1)
        constraints.append(u_hourly_green_sources >= 0)
        constraints.append(u_hourly_green_sources <= 1)
        constraints.append(x_hourly_nongreen >= 0)
        constraints.append(x_hourly_nongreen <= 1)
        constraints.append(x_hourly_green >= 0)
        constraints.append(x_hourly_green <= 1)
        constraints.append(x_chg >= 0)
        constraints.append(x_chg <= 1)
        constraints.append(x_dchg >= 0)
        constraints.append(x_dchg <= 1)
        
        # Power balance constraints 0: grid net power equals load minus PV plus battery operations
        if self.mode == 'prescient':
            constraints.append(Pnet == self.meas_PL - self.meas_PV + Pchg.reshape(-1,order='C') - Pdchg.reshape(-1,order='C'))
        elif self.mode == 'deterministic':
            constraints.append(Pnet == self.forecasted_PL - self.forecasted_PV + Pchg.reshape(-1, order='C') - Pdchg.reshape(-1, order='C'))
        
        # Power balance constraints 1:Pnet are all power required from both green and nongreen sources
        constraints.append(posPnet == Pgreen.sum(axis=1) + Pnongreen)
        
        # Power balance constaints 2:To relax the constaint, the linear inequality is exploited here
        constraints.append(Pnet == posPnet - negPnet)
        constraints.append(Pnet <= posPnet + negPnet)
        constraints.append(Pnet >= -(posPnet + negPnet))

        # Match hourly and time-step power variables
        for h in range(n_hourly_blocks):
            start_idx = h * n_points_1hour * self.weight_hourly_cons
            end_idx = start_idx + n_points_1hour * self.weight_hourly_cons
            # For nongreen power
            constraints.append(Pnongreen[start_idx:end_idx] == P_hourly_nongreen[h])
            # constraints.append(negPnet[start_idx:end_idx] == negPnet_hourly[h])
            constraints.append(x_nongreen[start_idx:end_idx] == x_hourly_nongreen[h]) # shape (length_optimvar,) element-wise
            constraints.append(u_green_sources[start_idx:end_idx] == u_hourly_green_sources[h]) # shape (length_optimvar, n_green_sources) element-wise
            # For green power
            for g in range(n_green_sources):
                constraints.append(Pgreen[start_idx:end_idx, g] == P_hourly_green[h, g])
                constraints.append(x_green[start_idx:end_idx, g] == x_hourly_green[h, g]) # shape (length_optimvar, n_green_sources) element-wise
        
        # Big-M constraints to link power variables with binary variables
        constraints.append(Pgreen <= M * x_green)  # shape (length_optimvar, n_green_sources) element-wise
        constraints.append(Pnongreen <= M * x_nongreen)  # shape (length

        EPSILON_POS: float = 1e-5  # Small constant to avoid numerical issues
        constraints.append(Pgreen >= EPSILON_POS * x_green)  # shape (length_optimvar, n_green_sources) element-wise
        constraints.append(Pnongreen >= EPSILON_POS * x_nongreen)  # shape
        
        # To force that at each time step, only one green source or one nongreen source can be used
        constraints.append(u_green_sources + x_nongreen <= 1)  # shape (length_optimvar,) element-wise
        constraints.append(x_green <= cp.reshape(u_green_sources, (length_optimvar, 1))) # shape (length_optimvar, n_green_sources) element-wise
        
        
        # Each imported green source must not exceed the green allocation quota which should convert to kWh in the optimization
        constraints.append(cp.sum(Pgreen, axis=0) * resolution_in_hour <= self.green_allocation_quota)
        
        # Battery charging and discharging rate limits
        constraints.append(Pchg<= self.battery.max_charge_rates * x_chg)
        constraints.append(Pdchg<= self.battery.max_discharge_rates * x_dchg)
        constraints.append(x_chg+x_dchg<=1)
        
        # Force smooth dischrage 

        constraints.append(Pdchg[1:, :] - Pdchg[:-1, :] <= self.smooth_charge_factor * self.battery.max_discharge_rates[0])
        constraints.append(Pdchg[1:, :] - Pdchg[:-1, :] >= -self.smooth_charge_factor * self.battery.max_discharge_rates[0])
        constraints.append(Pchg[1:, :] - Pchg[:-1, :] <= self.smooth_charge_factor * self.battery.max_charge_rates[0])
        constraints.append(Pchg[1:, :]- Pchg[:-1, :] >= -self.smooth_charge_factor * self.battery.max_charge_rates[0])
        
        # Battery SOC constraints
        constraints.append(soc[0, :] == self.battery.init_socs)  # initial SOC condition
        constraints.append(soc >= np.matlib.repmat(self.battery.min_socs, length_optimvar+1, 1))
        constraints.append(soc <= np.matlib.repmat(self.battery.max_socs, length_optimvar+1, 1))
        for j in range(no_batts):
            charge_factor = (self.battery.charge_effs[j] * 100 * resolution_in_hour / self.battery.capacity[j])
            discharge_factor = (resolution_in_hour * 100 / (self.battery.discharge_effs[j] * self.battery.capacity[j]))
            constraints.append(soc[1:, j] == soc[:-1, j] + charge_factor * Pchg[:, j] - discharge_factor * Pdchg[:, j])
            
        # End-of-day SOC constraints if terminal SOC targets are provided
        if self.terminal_socs is not None:
            # end_date_idx_list = [i * self.n_points_1day - 1 for i in range(1, n_horizon_days + 1)] # Consider end of day timestamp at 23:45 for 15-min resolution
            end_date_idx_list = [i * self.n_points_1day for i in range(1, n_horizon_days + 1)] # Consider end of day timestamp at 00:00 for 15-min resolution
            constraints.append(soc[end_date_idx_list, :] >= self.terminal_socs)
            constraints.append(soc[-1, :]>=self.terminal_socs)
            
            
        
        # Limit the number of days with nongreen usage
        for day in range(n_horizon_days):
            bh = int(day * n_hourly_blocks / n_horizon_days)

            eh = int((day + 1) * n_hourly_blocks / n_horizon_days)
            constraints += [
                x_hourly_nongreen[bh:eh] <= u_normXnongreen1day[day],
                # no need for the negative lower bound; you already bound binaries in [0,1]
            ]
            
        constraints.append(cp.sum(u_normXnongreen1day) <= n_horizon_days - self.n_RE_days)
        
        # Define the overall optimization problem
        prob = cp.Problem(cp.Minimize(total_obj), constraints)
        print(f"Total number of optimization variables: {sum(var.size for var in prob.variables())}")
        
        # Solve the problem using the specified solver, config that all constraints must be satisfied
        sol_status = prob.solve(solver=solver, IntFeasTol=1e-9, FeasibilityTol=1e-9, OptimalityTol=1e-9)
        print(sol_status, prob.status)
        # Assert if the solver reach optimal
        assert prob.status in ["optimal", "optimal_inaccurate"], f"Solver did not reach optimality, status: {prob.status}"
        assert np.all(Pnongreen.value <= M * x_nongreen.value + 1e-3), "Pnongreen exceeds Big-M limit"

        # Collect the solution into a DataFrame
        result_df = pd.DataFrame({
            'Datetime': self.target_dt,
            'measured_PL': self.meas_PL,
            'measured_PV': self.meas_PV,
            'measured_Netload': self.meas_PL - self.meas_PV,
            'forecasted_PL': self.forecasted_PL,
            'forecasted_PV':self.forecasted_PV,
            'forecasted_Netload': self.forecasted_PL - self.forecasted_PV,
            'Pnet_DA': Pnet.value,
            'posPnet_DA': posPnet.value,
            'negPnet_DA': negPnet.value} | 
            {f'Pgreen_UGT{idx+1}_DA': Pgreen.value[:, idx] for idx in range(n_green_sources)} |
            {'Pgreen_DA': np.sum(Pgreen.value, axis=1),}|
            {f'x_green_UGT{idx+1}_DA': x_green.value[:, idx] for idx in range(n_green_sources)} |
            {'x_green_DA' : np.max(x_green.value, axis=1)} |
            {'Pnongreen_DA': Pnongreen.value,
             'x_nongreen_DA': x_nongreen.value,
             'daily_u_normXnongreen1day': np.kron(u_normXnongreen1day.value.reshape(-1,), np.ones((self.n_points_1day))).flatten(),
             'x_chg': x_chg.value[:, 0].flatten(),
             'x_dchg': x_dchg.value[:, 0].flatten(),
             'Pdchg': np.sum(Pdchg.value, axis=1),
             'Pchg': np.sum(Pchg.value, axis=1),
             'soc': soc.value[:-1, 0],
             'Pnet_RT': (self.meas_PL - self.meas_PV + np.sum(Pchg.value, axis=1) - np.sum(Pdchg.value, axis=1)                      
                                  - np.sum(Pgreen.value, axis=1) - Pnongreen.value), # Calculate mismatch power in realtime
                'TOU_PLANNING_BUY_RATE': self.TOU_PLANNING_BUY_RATE,
                'TOU_PLANNING_SELL_RATE': self.TOU_PLANNING_SELL_RATE,
                'TOU_RT_BUY_RATE': self.TOU_RT_BUY_RATE,
                'TOU_RT_SELL_RATE': self.TOU_RT_SELL_RATE,} |
            {f"UGT{idx+1}_PLANNING_BUY_RATE": UGT_PLANNING_BUY_RATE_array[:, idx] for idx in range(n_green_sources)} |
            {f"UGT{idx+1}_PLANNING_SELL_RATE": UGT_PLANNING_SELL_RATE_array[:, idx] for idx in range(n_green_sources)} |
            {f"UGT{idx+1}_RT_BUY_RATE": UGT_RT_BUY_RATE_array[:, idx] for idx in range(n_green_sources)} |
            {f"UGT{idx+1}_RT_SELL_RATE": UGT_RT_SELL_RATE_array[:, idx] for idx in range(n_green_sources)}
            
        )
        EPSILON = 1e-3
        # Create status column stating whether nongreen imported in the day

        result_df['daily_nongreen_status'] = (result_df['Pnongreen_DA'].groupby(result_df['Datetime'].dt.date).transform('max') > EPSILON).astype(int)
        result_df['daily_green_status'] = 1 - result_df['daily_nongreen_status']
        print(f"the next step soc is {soc.value[-1, :]}")
        
        # Assert all the anticipated result the should hold all constraints
        
        # Power balance constraint must be hold
        assert np.allclose(result_df['Pnet_DA'], result_df['posPnet_DA'] - result_df['negPnet_DA'], atol=EPSILON), \
            f"Mismatch in Pnet_DA balance of Pnet_DA == posPnet_DA - negPnet_DA: max error={np.max(np.abs(result_df['Pnet_DA'] - (result_df['posPnet_DA'] - result_df['negPnet_DA'])))}"
        if self.mode == 'prescient':
            assert np.allclose(result_df['Pnet_DA'], result_df['measured_PL'] - result_df['measured_PV'] + result_df['Pchg'] - result_df['Pdchg'], atol=EPSILON), \
            f"Mismatch in Pnet_DA power balance of Pnet_DA == PL - PV + Pc - Pd: max error={np.max(np.abs(result_df['Pnet_DA'] - (result_df['measured_PL'] - result_df['measured_PV'] + result_df['Pchg'] - result_df['Pdchg'])))}"

        elif self.mode == 'deterministic':
            assert np.allclose(result_df['Pnet_DA'], result_df['forecasted_PL'] - result_df['forecasted_PV'] + result_df['Pchg'] - result_df['Pdchg'], atol=EPSILON), \
                f"Mismatch in Pnet_DA power balance of Pnet_DA == PL - PV + Pc - Pd: max error={np.max(np.abs(result_df['Pnet_DA'] - (result_df['forecasted_PL'] - result_df['forecasted_PV'] + result_df['Pchg'] - result_df['Pdchg'])))}"

        # Pnet constaints
        assert np.allclose(result_df['posPnet_DA'], result_df['Pgreen_DA'] + result_df['Pnongreen_DA'], atol=EPSILON), \
            f"Mismatch in posPnet_DA == accumulated_Pgreen_DA + Pnon_green_DA: max error={np.max(np.abs(result_df['posPnet_DA'] - (result_df['accumulated_Pgreen_DA'] + result_df['Pnongreen_DA'])))}"
            
        # Pnet lay between some of posPnet and negPnet
        assert np.all(result_df['Pnet_DA'] <= result_df['posPnet_DA'] + result_df['negPnet_DA'] + EPSILON), \
            "Pnet_DA should be less than or equal to posPnet_DA + negPnet_DA"
        assert np.all(result_df['Pnet_DA'] >= -(result_df['posPnet_DA'] + result_df['negPnet_DA']) - EPSILON), \
            "Pnet_DA should be greater than or equal to -(posPnet_DA + negPnet_DA)"
            
        # Check the binary status variables there is no impoting both green and nongreen power at the same time
        assert np.all(result_df['x_nongreen_DA'] + result_df['x_green_DA'] <= 1 + EPSILON), \
            "At each time step, the sum of x_nongreen_DA and x_green_DA should be less than or equal to 1"
        
        # assert smooth battery discharge
        assert np.all(np.abs(result_df['Pdchg'].diff().fillna(0)) <= self.smooth_charge_factor * self.battery.max_discharge_rates[0] + EPSILON), f"Smooth battery discharge constraint violated: max change={np.max(np.abs(result_df['Pdchg'].diff().fillna(0)))}"
        # Adjust the results dataframe to include RT variables
        return prob, result_df, prob.value, prob.status
    
            
    
    def _stochastic_optimization(self, solver=cp.GUROBI):
        '''
        Only one green source and one nongreen source is considered here
        
        '''
        length_optimvar = len(self.target_dt)
        
        resolution_in_hour = (self.target_dt[1] - self.target_dt[0]).seconds / 3600 # 0.25 for 15-min resolution

        n_horizon_days = self.n_horizon_days
        n_horizon_hours = math.ceil((self.target_dt[-1]- self.target_dt[0]).total_seconds() / 3600)
        n_green_sources = self.n_green_sources
        n_realizations = self.n_realizations
        
        n_points_1hour = int(1 / resolution_in_hour)
        n_hourly_blocks = int(n_horizon_hours/self.weight_hourly_cons)
        n_hours_1day = 24
        replication_vector = np.ones((n_points_1hour * self.weight_hourly_cons, 1))
        replication_row  = np.ones((1, n_points_1hour * self.weight_hourly_cons))
        
        # Define decision variables
        # 1st-decision variables
        P_hourly_green_DA = cp.Variable((n_hourly_blocks, 1), nonneg=True, name='P_hourly_green_DA')
        Pgreen_DA = cp.Variable((length_optimvar,), nonneg=True, name='Pgreen_DA')
        P_hourly_nongreen_DA = cp.Variable((n_hourly_blocks, 1), nonneg=True, name='P_hourly_nongreen_DA')
        Pnongreen_DA = cp.Variable((length_optimvar,), nonneg=True, name='Pnongreen_DA')
        
        
        # P_hourly_sell_DA = cp.Variable((n_hourly_blocks, 1), nonneg=True, name='P_hourly_sell_DA')
        # Psell_DA = cp.Variable((length_optimvar,), nonneg=True, name='Psell_DA')
        
        
        

        # just try to create full decision variables without hourly reduction
        # Pgreen_DA = cp.Variable(length_optimvar, nonneg=True, name='Pgreen_DA')
        # Pnongreen_DA = cp.Variable(length_optimvar, nonneg=True, name='Pnongreen_DA')
        # Psell_DA = cp.Variable((length_optimvar,), nonneg=True, name='Psell_DA')
        
        # Battery
        no_batts = self.battery.no_batts
        Pdchg = cp.Variable((length_optimvar, no_batts), nonneg=True, name='Pdchg')
        Pchg  = cp.Variable((length_optimvar, no_batts), nonneg=True, name='Pchg')
        soc = cp.Variable((length_optimvar + 1, no_batts), name='soc')
        x_chg   = cp.Variable((length_optimvar, no_batts), integer=True, name='x_chg')
        x_dchg    = cp.Variable((length_optimvar, no_batts), integer=True, name='x_dchg')
        
        # Integer decision variables for grid status (using Big-M formulation later)
        u_normXnongreen1day = cp.Variable(n_horizon_days, integer=True, name='normXnongreen1Day')
        x_hourly_green = cp.Variable(n_hourly_blocks, integer=True, name='x_hourly_green')
        x_green = cp.Variable(length_optimvar, integer=True, name='x_green')
        x_hourly_nongreen = cp.Variable(n_hourly_blocks, integer=True, name='x_hourly_nongreen')
        x_nongreen = cp.Variable(length_optimvar, integer=True, name='x_nongreen')
    
                
        # 2nd stage variables
        Pgreen_RT = cp.Variable((n_realizations, length_optimvar), nonneg=True, name='Pgreen_RT')
        Pnongreen_RT = cp.Variable((n_realizations, length_optimvar), nonneg=True, name='Pnongreen_RT')
        Pcurtailed_RT = cp.Variable((n_realizations,length_optimvar), nonneg=True, name='Pcurtailed_RT')
        # Psell_RT = cp.Variable((n_realizations, length_optimvar), nonneg=True, name='Psell_RT')
        # Get the buy sell rates in array format for optimization of shape (length_optimvar, n_green_sources)
        UGT_PLANNING_BUY_RATE_array = np.zeros((length_optimvar, n_green_sources))
        UGT_PLANNING_SELL_RATE_array = np.zeros((length_optimvar, n_green_sources))
        UGT_RT_BUY_RATE_array = np.zeros((length_optimvar, n_green_sources))
        UGT_RT_SELL_RATE_array = np.zeros((length_optimvar, n_green_sources))
        for source_idx in range(n_green_sources):
            UGT_PLANNING_BUY_RATE_array[:, source_idx] = self.UGT_PLANNING_BUY_RATE[f'UGT{source_idx+1}']
            UGT_PLANNING_SELL_RATE_array[:, source_idx] = self.UGT_PLANNING_SELL_RATE[f'UGT{source_idx+1}']
            UGT_RT_BUY_RATE_array[:, source_idx] = self.UGT_RT_BUY_RATE[f'UGT{source_idx+1}']
            UGT_RT_SELL_RATE_array[:, source_idx] = self.UGT_RT_SELL_RATE[f'UGT{source_idx+1}']
        print()
        
        weight_stochastic = 1
        obj_DA = (UGT_PLANNING_BUY_RATE_array.T @ Pgreen_DA + 
                  self.TOU_PLANNING_BUY_RATE.T @ Pnongreen_DA)
        obj_RT = weight_stochastic * 1/ n_realizations * (cp.sum(cp.multiply(Pgreen_RT, UGT_RT_BUY_RATE_array.T)) +  # Broadcast the buy cost to (n_realizations, length_optimvar) then multiply element-wise
                                cp.sum(cp.multiply(Pnongreen_RT, self.TOU_RT_BUY_RATE)) 
                                )  # Broadcast the buy cost to (n_realizations, length_optimvar) then multiply element-wise

        total_obj = (obj_DA + obj_RT)* resolution_in_hour

        
        M: float = 1e2  # Big-M constant for constraints
    
        # Initialize constraints list
        constraints = []
        
        # Integer variable constraints (ensuring binary values 0 or 1)
        constraints.append(u_normXnongreen1day >= 0)
        constraints.append(u_normXnongreen1day <= 1)
        constraints.append(x_hourly_green >= 0)
        constraints.append(x_hourly_green <= 1)
        constraints.append(x_hourly_nongreen >= 0)
        constraints.append(x_hourly_nongreen <= 1)
        constraints.append(x_dchg >= 0)
        constraints.append(x_dchg <= 1)
        constraints.append(x_chg >= 0)
        constraints.append(x_chg <= 1)

        #Power balance constraint: grid net power equals load minus PV plus battery operations
        power_balance_LHS = (cp.reshape(Pgreen_DA, (1, -1), order='C') + cp.reshape(Pnongreen_DA, (1, -1), order='C') + 
                                Pgreen_RT + Pnongreen_RT +cp.reshape(Pdchg, (1, -1), order='C') + self.pv_realizations)
        power_balance_RHS = (cp.reshape(Pchg, (1, -1), order='C') + Pcurtailed_RT + self.pl_realizations)
        # # ADD EXACTLY ONE equality:
        constraints.append(power_balance_LHS == power_balance_RHS)

        # sanity
        print("Power balance constraint shapes:", power_balance_LHS.shape, power_balance_RHS.shape)
            
        
        
        # Map hourly DA variables to full length variables
        for h in range(n_hourly_blocks):
            start_idx = h * n_points_1hour * self.weight_hourly_cons
            end_idx = start_idx + n_points_1hour * self.weight_hourly_cons
            constraints.append(Pgreen_DA[start_idx:end_idx] == P_hourly_green_DA[h])
            constraints.append(Pnongreen_DA[start_idx:end_idx] == P_hourly_nongreen_DA[h])
            # constraints.append(Psell_DA[start_idx:end_idx] == P_hourly_sell_DA[h])
            constraints.append(x_green[start_idx:end_idx] == x_hourly_green[h])
            constraints.append(x_nongreen[start_idx:end_idx] == x_hourly_nongreen[h])
        
        # Big-M constraints to link power variables with binary variables
        constraints.append(Pgreen_DA <= M * x_green)
        constraints.append(Pnongreen_DA <= M * x_nongreen)


        constraints.append(Pgreen_RT <= M * x_green.reshape((1, length_optimvar), order='C')) # sum along the green source axis to get shape (n_realizations, length_optimvar)
        constraints.append(Pnongreen_RT <= M * x_nongreen.reshape((1, length_optimvar), order='C'))
        
        # Impose strictly import either green or nongreen at each time step
        EPSILON_POSITIVE = 1e-5
        constraints.append(Pgreen_DA >= EPSILON_POSITIVE * x_green)
        constraints.append(Pnongreen_DA >= EPSILON_POSITIVE * x_nongreen)
        constraints.append(Pgreen_RT >= EPSILON_POSITIVE * x_green.reshape((1, length_optimvar), order='C'))
        constraints.append(Pnongreen_RT >= EPSILON_POSITIVE * x_nongreen.reshape((1, length_optimvar), order='C'))


        # To force that at each time step, only one green source or one nongreen source can be used
        constraints.append(x_green + x_nongreen <= 1)
        
        # # Battery charging and discharging rate limits
        constraints.append(Pchg<= self.battery.max_charge_rates * x_chg)    
        constraints.append(Pdchg<= self.battery.max_discharge_rates * x_dchg)   
        constraints.append(x_chg+x_dchg<=1)

        # Force smooth dischrage 
        constraints.append(Pdchg[1:, :] - Pdchg[:-1, :] <= self.smooth_charge_factor * self.battery.max_discharge_rates)
        constraints.append(Pdchg[1:, :] - Pdchg[-1:, :] >= -self.smooth_charge_factor * self.battery.max_discharge_rates)
        constraints.append(Pchg[1:, :] - Pchg[:-1, :] <= self.smooth_charge_factor * self.battery.max_charge_rates)
        constraints.append(Pchg[1:, :] - Pchg[-1:, :] >= -self.smooth_charge_factor * self.battery.max_charge_rates)
        
         # Battery state-of-charge (SOC) dynamics constraints
        constraints.append(soc[0, :] == self.battery.init_socs)  # initial SOC condition
        constraints.append(soc >= np.matlib.repmat(self.battery.min_socs, length_optimvar+1, 1))
        constraints.append(soc <= np.matlib.repmat(self.battery.max_socs, length_optimvar+1, 1))
        for j in range(no_batts):
            charge_factor = self.battery.charge_effs[j] * 100 * resolution_in_hour / self.battery.capacity[j]
            discharge_factor = (100 * resolution_in_hour / (self.battery.discharge_effs[j] * self.battery.capacity[j]))
            constraints.append(soc[1:, j] == soc[:-1, j] + charge_factor * Pchg[:, j] - discharge_factor * Pdchg[:, j])
        
       # End-of-day SOC constraints if terminal SOC targets are provided
        if self.terminal_socs is not None:
            # end_date_idx_list = [i * self.n_points_1day - 1 for i in range(1, n_horizon_days + 1)] # Consider end of day timestamp at 23:45 for 15-min resolution
            end_date_idx_list = [i * self.n_points_1day for i in range(1, n_horizon_days + 1)] # Consider end of day timestamp at 00:00 for 15-min resolution
            constraints.append(soc[end_date_idx_list, :] >= self.terminal_socs)
            constraints.append(soc[-1, :]>=self.terminal_socs)

        # Limit the number of days with nongreen usage
        for day in range(n_horizon_days):
            bh = int(day * n_hourly_blocks / n_horizon_days)

            eh = int((day + 1) * n_hourly_blocks / n_horizon_days)
            constraints += [
                x_hourly_nongreen[bh:eh] <= u_normXnongreen1day[day],
                # no need for the negative lower bound; you already bound binaries in [0,1]
            ]

            
        constraints.append(cp.sum(u_normXnongreen1day) <= n_horizon_days - self.n_RE_days)


        
        # Define the overall optimization problem
        # RE100EMS.min_violation_problem(total_obj, constraints, solver=solver, verbose=True)
        prob = cp.Problem(cp.Minimize(total_obj), constraints)
        print("Is DCP:", prob.is_dcp())
        print("Is DPP:", prob.is_dpp())
        # print(prob.variables())
        # To get number of variables:
        total_vars = sum(var.size for var in prob.variables())
        print("Total number of optimization variables:", total_vars)
        # Solve the problem using the specified solver

        sol_status = prob.solve(solver=solver, save_file='stochastic_optimization.lp',)
        # Assert if the solver reach optimal
        assert prob.status in ["optimal", "optimal_inaccurate"], f"Solver did not reach optimality, status: {prob.status}"
        # assert np.all(Pnongreen_DA.value <= M * x_nongreen.value + 1e-3), "Pnongreen exceeds Big-M limit"
        print(f"normXnongreen1day values: {u_normXnongreen1day.value}")
         # Collect the solution into a DataFrame
        result_df = pd.DataFrame({
            'Datetime': self.target_dt,
            'measured_PL': self.meas_PL,
            'measured_PV': self.meas_PV,
            'measured_Netload': self.meas_PL - self.meas_PV,
            'forecasted_PL': self.forecasted_PL,
            'forecasted_PV': self.forecasted_PV,
            'forecasted_Netload': self.forecasted_PL - self.forecasted_PV,
            'Pgreen_DA': Pgreen_DA.value,
            'Pnongreen_DA': Pnongreen_DA.value,
            # 'Psell_DA': Psell_DA.value,
            'x_green_DA': x_green.value.flatten(),
            'x_nongreen_DA': x_nongreen.value.flatten(),
            'daily_u_normXnongreen1day': np.kron(u_normXnongreen1day.value, np.ones((self.n_points_1day))), 
            'x_chg': x_chg.value[:, 0].flatten(),
            'x_dchg': x_dchg.value[:, 0].flatten(),
            'Pdchg': np.sum(Pdchg.value, axis=1),
            'Pchg': np.sum(Pchg.value, axis=1),
            'soc': soc.value[:-1, 0],
            'Pnet_RT': (self.meas_PL - self.meas_PV + np.sum(Pchg.value, axis=1) - np.sum(Pdchg.value, axis=1)                      
                                 - Pgreen_DA.value - Pnongreen_DA.value), # Calculate mismatch power in realtime
            'TOU_PLANNING_BUY_RATE': self.TOU_PLANNING_BUY_RATE,
            'TOU_PLANNING_SELL_RATE': self.TOU_PLANNING_SELL_RATE,
            'TOU_RT_BUY_RATE': self.TOU_RT_BUY_RATE,
            'TOU_RT_SELL_RATE': self.TOU_RT_SELL_RATE,} |
            {f"UGT{idx+1}_PLANNING_BUY_RATE": UGT_PLANNING_BUY_RATE_array[:, idx] for idx in range(n_green_sources)} |
            {f"UGT{idx+1}_PLANNING_SELL_RATE": UGT_PLANNING_SELL_RATE_array[:, idx] for idx in range(n_green_sources)} |
            {f"UGT{idx+1}_RT_BUY_RATE": UGT_RT_BUY_RATE_array[:, idx] for idx in range(n_green_sources)} |
            {f"UGT{idx+1}_RT_SELL_RATE": UGT_RT_SELL_RATE_array[:, idx] for idx in range(n_green_sources)}
        )

        if self.extract_realizations is not None:
            for r in self.extract_realizations:
                result_df[f'Pgreen_RT_{r}'] = Pgreen_RT.value[r, :]
                result_df[f'Pnongreen_RT_{r}'] = Pnongreen_RT.value[r, :]
                result_df[f'Pcurtailed_RT_{r}'] = Pcurtailed_RT.value[r, :]
        
         # Create status column stating whether nongreen imported in the day
            
        EPSILON = 1e-3
        result_df['daily_nongreen_status'] = (result_df['Pnongreen_DA'].groupby(result_df['Datetime'].dt.date).transform('max') > EPSILON).astype(int)
        result_df['daily_green_status'] = 1 - result_df['daily_nongreen_status']
        print(f"the next step soc is {soc.value[-1, :]}")

        # adjusted_result_df, daily_status_df = RE100EMS._adjust_RT_results_df(result_df)
        
        # assert np.allclose(adjusted_result_df['posPnet_RT'], adjusted_result_df['Pgreen_RT'] + adjusted_result_df['Pnongreen_RT'], atol=EPSILON), \
        #     f"Mismatch in posPnet_RT and importing green and nongreen power with positive Pnet_RT {adjusted_result_df['posPnet_RT'].sum()} and Pgreen_RT {adjusted_result_df['Pgreen_RT'].sum()} and Pnongreen_RT {adjusted_result_df['Pnongreen_RT'].sum()}"

        # assert np.allclose(adjusted_result_df['negPnet_RT'], adjusted_result_df['Psell_RT'], atol=EPSILON), \
        #     f"Mismatch in negPnet_RT and selling power"

        # assert np.allclose(adjusted_result_df['Pnet_RT'], adjusted_result_df['Pgreen_RT'] + adjusted_result_df['Pnongreen_RT'] - adjusted_result_df['Psell_RT'], atol=EPSILON), \
        #     f"Mismatch in Pnet_RT considering import and export Power"

        # assert np.allclose(adjusted_result_df['Pnet_RT'], adjusted_result_df['posPnet_RT'] - adjusted_result_df['negPnet_RT'], atol=EPSILON), \
        #     f"Mismatch in Pnet_RT balance"
            
        exit_flag: str = prob.status
        fval: float = prob.value
        return prob, result_df, fval, exit_flag
        
    
    def optimize(self, solver=cp.GUROBI,):
    
        if self.mode in ['prescient','deterministic']:
            return self._deterministic_optimization_multi_providers(solver=solver)
            
        elif self.mode == 'stochastic':
            return self._stochastic_optimization(solver=solver)
            
    @staticmethod
    def _adjust_RT_results_df(result_df, resolution_in_hour=1/4, n_green_sources=3):
        epsilon = 1e-3
        adjusted_results_df = result_df.copy()
        
        adjusted_results_df['posPnet_RT'] = adjusted_results_df['Pnet_RT'].clip(lower=0)
        adjusted_results_df['negPnet_RT'] = (-adjusted_results_df['Pnet_RT']).clip(lower=0)
        

        export_RT_cond = (adjusted_results_df['Pnet_RT'] < 0)
        adjusted_results_df['Psell_RT'] = np.where(export_RT_cond, adjusted_results_df['negPnet_RT'], 0)
        
        

    @staticmethod
    def adjust_RT_results_df(result_df, n_points_1hour=4, mode='deterministic'):
        epsilon = 1e-3
        resolution_in_hour = 1 / n_points_1hour
        adjusted_results_df = result_df.copy()
        
        # Calculate posPnet_RT and negPnet_RT
        adjusted_results_df['posPnet_RT'] = adjusted_results_df['Pnet_RT'].clip(lower=0)
        adjusted_results_df['negPnet_RT'] = (-adjusted_results_df['Pnet_RT']).clip(lower=0)
        
        # Export all excess power when Pnet_RT < 0 at real-time
        export_RT_cond = (adjusted_results_df['Pnet_RT'] < 0)
        adjusted_results_df['Pcurtailed_RT'] = np.where(export_RT_cond, adjusted_results_df['negPnet_RT'], 0)
        
        # import green and nongreen power based on planned daily status
        import_green_RT_cond = (adjusted_results_df['Pnet_RT'] > epsilon)&(adjusted_results_df['daily_green_status']>epsilon)
        import_nongreen_RT_cond = (adjusted_results_df['Pnet_RT'] > epsilon)&(adjusted_results_df['daily_nongreen_status']>epsilon)
        adjusted_results_df['Pgreen_RT'] = np.where(import_green_RT_cond, adjusted_results_df['posPnet_RT'], 0)
        adjusted_results_df['Pnongreen_RT'] = np.where(import_nongreen_RT_cond, adjusted_results_df['posPnet_RT'], 0)
        

        # nongreen imported power cost calculation
        adjusted_results_df['cost_buying_nongreen_DA'] = (adjusted_results_df['Pnongreen_DA'] * adjusted_results_df['TOU_PLANNING_BUY_RATE'])* resolution_in_hour
        adjusted_results_df['cost_buying_nongreen_RT'] = (adjusted_results_df['Pnongreen_RT'] * adjusted_results_df['TOU_RT_BUY_RATE']) * resolution_in_hour

        # green imported power cost calculation, note that only one green source in the rolling horizon is considered here
        adjusted_results_df['cost_buying_green_DA'] = (adjusted_results_df['Pgreen_DA'] * adjusted_results_df['UGT1_PLANNING_BUY_RATE']) * resolution_in_hour
        adjusted_results_df['cost_buying_green_RT'] = (adjusted_results_df['Pgreen_RT'] * adjusted_results_df['UGT1_RT_BUY_RATE']) * resolution_in_hour

        # selling revenue calculation
        # if mode == 'prescient':, unable selling mode
        #     adjusted_results_df['cost_selling_DA'] = (adjusted_results_df['negPnet_DA'] * adjusted_results_df['UGT1_PLANNING_SELL_RATE']) * resolution_in_hour
        # elif mode == 'deterministic':
        #     adjusted_results_df['cost_selling_DA'] = (adjusted_results_df['negPnet_DA'] * adjusted_results_df['UGT1_PLANNING_SELL_RATE']) * resolution_in_hour
        # elif mode == 'stochastic':
        #     adjusted_results_df['cost_selling_DA'] = (adjusted_results_df['Psell_DA'] * adjusted_results_df['UGT1_PLANNING_SELL_RATE']) * resolution_in_hour
        
        # adjusted_results_df['cost_selling_RT'] = (adjusted_results_df['Psell_RT'] * adjusted_results_df['UGT1_RT_SELL_RATE']) * resolution_in_hour
        
        # net cost calculation
        adjusted_results_df['net_cost_DA'] = (adjusted_results_df['cost_buying_nongreen_DA'] + adjusted_results_df['cost_buying_green_DA'])
        adjusted_results_df['net_cost_RT'] = (adjusted_results_df['cost_buying_nongreen_RT'] + adjusted_results_df['cost_buying_green_RT'])
        adjusted_results_df['net_cost'] = adjusted_results_df['net_cost_DA'] + adjusted_results_df['net_cost_RT']
        
        # cumulative net cost calculation
        adjusted_results_df['cumsum_net_cost'] = adjusted_results_df['net_cost'].cumsum()
    
        return adjusted_results_df