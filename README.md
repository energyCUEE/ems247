# Stochastic EMS for Optimal 24/7 Carbon-Free Energy Operations
This repository contains the code and resources for the EMS designed to optimize 24/7 carbon-free energy operations. Note that, the variables name were created at the time when the term *RE days* was referred to *green days* which is now updated to *CF (Carbon-free) days* in the paper.

## Folder Structure
- [data](./data/): Contains measurement of load and solar (irradiance) data which index is planning date.
- [selected_models](./selected_models/): Contains best results from three different models: LGBM, LSTM, NNprophet for both load and solar forecasting.
- [src](./src/): Contains EMS source codes for which deterministic and stochastic optimization are implemented.
- [configs](./configs/): Contains configuration files for different experiments.
- [mat](./mat/): Contains mean and covariance matrices for stochastic formulation.

## Experiment codes
The following scripts could be used to generate results presented in the paper.
### One batch experiments
1. for prescient and deterministic EMS of one green source we use :
```bash
python deterministic_one_batch_vary_RE.py --solve_mode "optimize" --planning_date "2024-08-02" --optimization_configs_path "./configs/deterministic_one_batch_configs.json" --saved-path "./results" 
```
This command runs experiments for one batch with varying anticipated RE days,
which use `deterministic_one_batch_vary_RE.py` and `deterministic_one_batch_vary_RE_nosell.py` for with and without energy selling cases, respectively.
2. for prescient and deterministic EMS of more than one green source, RE_days fixed, we use :
```bash
python deterministic_one_batch_multi_providers.py --solve_mode "optimize" --planning_date 2024-09-15 --n_RE_days 5 "--optimization_configs_path" "./configs/deterministic_one_batch_3providers_configs.json" --saved_path "./results"
```
which use `deterministic_one_batch_multi_providers.py` and `deterministic_one_batch_multi_providers_nosell.py` for with and without energy selling cases, respectively.
### Rolling experiments
For all formulations, which are prescient, deterministic, and stochastic, we use :
```bash
python rolling_opt.py --optimization_configs_path './configs/stochastic_rolling_configs.json' --saved_path './results/stochastic_rolling' > "./logs/stochastic_rolling_$(date +'%Y%m%d_%H%M%S').log" 2>&1
```
This command runs rolling experiments using `rolling_opt.py` and `rolling_opt_nosell.py` with different configuration files for each formulation.


