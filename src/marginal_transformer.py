import numpy as np
from scipy.stats import rankdata

class MarginalTransformer:
    
    def __init__(self):
        self.sorted_data = None
        self.uniform_grid = None
        self.n_samples = None
        self.n_dim = None

    def fit_transform(self, original_data):
        """Fit the marginal distributions and convert data to uniform (0, 1)

        pseudo-observations using native axis=0 ranking.
        """
        self.n_samples, self.n_dim = original_data.shape

        # 1. Store pre-sorted values along each dimension (54 columns)
        self.sorted_data = np.sort(original_data, axis=0)

        # 2. Define uniform grid corresponding to ranks (1 to N) / (N + 1)
        self.uniform_grid = np.linspace(
            1.0 / (self.n_samples + 1.0),
            self.n_samples / (self.n_samples + 1.0),
            self.n_samples,
        )

        # 3. Native vectorized ranking along columns (axis=0)
        u = rankdata(original_data, axis=0) / (self.n_samples + 1.0)

        return u

    def inverse_transform(self, u_simulated):
        """Invert simulated uniform values (0, 1) back to original data scale

        WITHOUT needing the original raw dataset array!
        """
        n_sim_samples, n_dim = u_simulated.shape
        scenarios = np.zeros_like(u_simulated)

        # Apply np.interp independently across each dimension
        # This replaces np.quantile (inverse of CDF) to avoid carrying historical data
        for col in range(n_dim):
            # np.interp(x_new, x_grid, y_grid)
            scenarios[:, col] = np.interp(
                x=u_simulated[:, col],
                xp=self.uniform_grid,  # Pre-defined uniform probabilities (0, 1)
                fp=self.sorted_data[:, col],  # Pre-sorted original values
                left=self.sorted_data[
                    0, col
                ],  # Clamp lower bounds to min observed
                right=self.sorted_data[
                    -1, col
                ],  # Clamp upper bounds to max observed
            )

        return scenarios