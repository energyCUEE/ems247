# Stochastic EMS for Optimal 24/7 Carbon-Free Energy Operations
This repository contains the code and resources for the EMS designed to optimize 24/7 carbon-free energy operations. Note that the variable names were created when the term *RE days* referred to *green days*, which is now updated to *CF (Carbon-free) days* in the paper.

## Folder Structure
- [data](./data/): Measurements of load and solar (irradiance) data.
- [selected_models](./selected_models/): Point forecasts (7-day ahead, 15-min resolution) from the best irradiance and load forecasting models, indexed by planning date.
- [src](./src/): EMS source code.
  - `re100_sp.py`: prescient, deterministic and two-stage stochastic formulations with CF-block constraints and a soft green-purchase budget.
  - `copula.py`, `marginal_transformer.py`: empirical-marginal + vine-copula model used to generate PV and load forecast-error scenarios for the stochastic formulation.
- [configs](./configs/): Configuration files for the experiments.
- [mat](./mat/): Error-model artifacts; `mat/copula/` holds the fitted PV and load copula pipelines used by the stochastic formulation.

## Environment
The code uses `cvxpy` with the Gurobi solver (a valid Gurobi license is required), plus `pandas`, `numpy`, `matplotlib`, `joblib` and `pyvinecopulib` (for the copula scenarios). Run all commands from the repository root.

## Experiment codes
### One batch experiments
1. Prescient or deterministic EMS with one green source, varying the anticipated CF days from 1 to 7:
```bash
python deterministic_one_batch_vary_RE.py --solve_mode "optimize" --planning_date "2024-08-02" --optimization_configs_path "./configs/deterministic_one_batch_configs.json" --saved_path "./results/deterministic_one_batch_vary_RE"
```
2. Prescient or deterministic EMS with three green sources (each with its own quota) and a fixed number of CF days:
```bash
python deterministic_one_batch_multi_providers.py --solve_mode "optimize" --planning_date "2024-09-15" --n_RE_days 5 --optimization_configs_path "./configs/deterministic_one_batch_3providers_configs.json" --saved_path "./results/deterministic_one_batch_multi_providers"
```
3. Two-stage stochastic EMS with one green source, varying the anticipated CF days. Forecast-error scenarios come from the copula model in `mat/copula/`:
```bash
python stochastic_one_batch_vary_RE.py --solve_mode "optimize" --planning_date "2024-08-02" --optimization_configs_path "./configs/stochastic_one_batch_configs.json" --saved_path "./results/stochastic_one_batch_vary_RE"
```
The config matches the stochastic setup of the rolling experiments: 20 scenarios, a soft green budget with `green_purchase_volume_factor` 0.7, and the same battery. Rough running time on a 32-core machine: about 1–1.5 min per anticipated-CF-days value (a few seconds for 7 CF days), so about 8 min for the full 1–7 sweep. Use e.g. `--n_RE_days 3 5` to solve only some values (the cost bar chart is drawn only for a full 1..N sweep). The saved results include the expected cost (`expected_cost`), the real-time recourse on measured data (`Pgreen_RT`, `Pnongreen_RT`, `Psell_RT`), and the real-time decisions of every scenario (`Pgreen_scenario_RT_{r}`, `Pnongreen_scenario_RT_{r}`, `Psell_scenario_RT_{r}`). The operation plots show one scenario's real-time decisions as dotted lines; pick it with `--plot_scenario` (default 0), which also works with `--solve_mode "plot"`.

For all scripts, `--solve_mode "plot"` re-creates the figures from previously saved results.

### Rolling experiments
`rolling_opt.py` runs a one-day rolling-horizon EMS over the planning dates in the config. One run uses one formulation, set by `mode` in the config (`prescient`, `deterministic` or `stochastic`), and sweeps the targeted CF days per horizon (`--n_RE_days`, default 1 to 7):
```bash
mkdir -p logs && python rolling_opt.py --optimization_configs_path "./configs/stochastic_rolling_configs.json" --saved_path "./results/stochastic_rolling" > "./logs/stochastic_rolling_$(date +'%Y%m%d_%H%M%S').log" 2>&1
```
Use `./configs/prescient_rolling_configs.json` or `./configs/deterministic_rolling_configs.json` for the other formulations. To run a subset, pass e.g. `--n_RE_days 3 5` or `--start_planning_date 2024-08-01 --end_planning_date 2024-08-31`. Separate runs with different `--n_RE_days` values can share a `--saved_path` and run in parallel.

On each planning date:
1. The green purchase is a soft budget equal to `green_purchase_volume_factor` x (forecasted load energy over the horizon). Each kWh above the budget is penalized by `weight_green_slack`. Set `green_purchase_volume_factor` to `null` to remove the budget.
2. After a 6-day warm-up, the first day's non-green status is fixed by an exponentially weighted vote of today's solution and the previous plans for that day, and the problem is re-solved.
3. Only the first day is executed. Its end-of-day SoC becomes the initial SoC for the next planning date, and real-time mismatches are settled with the measured data.

Outputs per targeted CF-day value `n`:
- `one_day_rolling_result_{n}REdays.csv`: executed (first-day) results, including day-ahead and real-time energy and `net_cost`.
- `collected_rolling_result_{n}REdays.csv`: full-horizon solutions for every planning date.
