import sys
from src import marginal_transformer
from src.marginal_transformer import MarginalTransformer
import joblib
import numpy as np

# The pickled pipelines reference the top-level module name 'marginal_transformer'
sys.modules.setdefault("marginal_transformer", marginal_transformer)

class DailyCopulaErrorModel:
    """
    Wraps a fitted one-day empirical-marginal + copula model for a single
    quantity's (PV or load) forecast error, at 96 15-min slots/day.
    """
    def __init__(self, 
                 pv_pipeline_path: str,
                 load_pipeline_path: str):
        self.full_time_idx = np.arange(96)  # Assuming 15-minute intervals for a full day
        self.pv_pipeline = joblib.load(pv_pipeline_path)
        self.pv_empCDF = self.pv_pipeline['marginal_transformer']
        self.pv_copula_model = self.pv_pipeline['copula_model']
        self.pv_time_idx = self.pv_pipeline['time_idx']
        
        self.load_pipeline = joblib.load(load_pipeline_path)
        self.load_empCDF = self.load_pipeline['marginal_transformer']
        self.load_copula_model = self.load_pipeline['copula_model']
        self.load_time_idx = self.load_pipeline['time_idx']

    def get_pv_error_scenarios(self, num_scenarios: int, n_horizon_days: int=7, base_seed: int=42):
        """
        Generate PV error scenarios using the fitted copula model.
        """
        # create a unique seed for each day to ensure different scenarios across days
        all_seeds = [base_seed + i*base_seed for i in range(n_horizon_days)]
        collected_arrays = []
        for seed in all_seeds:
            u_simulated = self.pv_copula_model.simulate(num_scenarios, seeds=[seed])
            day_time_pv_error_scenarios = self.pv_empCDF.inverse_transform(u_simulated)
            full_day_pv_error_scenarios = np.zeros((num_scenarios, len(self.full_time_idx)))
            full_day_pv_error_scenarios[:, self.pv_time_idx] = day_time_pv_error_scenarios
            collected_arrays.append(full_day_pv_error_scenarios)
        # concatenate all days' scenarios into a single array, 1D
        pv_error_scenarios = np.concatenate(collected_arrays, axis=1)
        return pv_error_scenarios

    def get_load_error_scenarios(self, num_scenarios: int, n_horizon_days: int=7, base_seed: int=42):
        """
        Generate load error scenarios using the fitted copula model.
        """
        # create a unique seed for each day to ensure different scenarios across days
        all_seeds = [base_seed + i*base_seed +10000 for i in range(n_horizon_days)] # offset by 10000 to avoid overlap with PV seeds
        collected_arrays = []
        for seed in all_seeds:
            u_simulated = self.load_copula_model.simulate(num_scenarios, seeds=[seed])
            # load already has 96 time slots, so no need to fill a larger array
            load_error_scenarios = self.load_empCDF.inverse_transform(u_simulated)
            collected_arrays.append(load_error_scenarios)
        # concatenate all days' scenarios into a single array, 1D
        load_error_scenarios = np.concatenate(collected_arrays, axis=1)
        return load_error_scenarios