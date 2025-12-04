import pandas as pd
import numpy as np

# For particle-spin hypothesis test
# Parameter values under each hypothesis
params = {}
params['mu_x1_H0'] = 0
params['sigma_x1_H0'] = 1
params['mu_x2_H0'] = 0
params['sigma_x2_H0'] = 1
params['rho_x1x2_H0'] = 0.5
params['mu_x1_H1'] = 0.25
params['sigma_x1_H1'] = 1
params['mu_x2_H1'] = 0
params['sigma_x2_H1'] = 1.5
params['rho_x1x2_H1'] = 0.
params['lambda_x3_H1'] = np.pi/2
params['amp_x3_H1'] = 0.3

# Generate synthetic data based on hypothesis
def run_simulation(N, hypothesis=None):
    if hypothesis == 'H0':
        mean = [params['mu_x1_H0'], params['mu_x2_H0']]
        cov = [[params['sigma_x1_H0']**2, params['rho_x1x2_H0']*params['sigma_x1_H0']*params['sigma_x2_H0']],
               [params['rho_x1x2_H0']*params['sigma_x1_H0']*params['sigma_x2_H0'], params['sigma_x2_H0']**2]]
        data_2d = np.random.multivariate_normal(mean, cov, N)
        x1 = data_2d[:, 0]
        x2 = data_2d[:, 1]
        x3 = np.random.uniform(-np.pi, np.pi, N)
    elif hypothesis == 'H1':
        mean = [params['mu_x1_H1'], params['mu_x2_H1']]
        cov = [[params['sigma_x1_H1']**2, params['rho_x1x2_H1']*params['sigma_x1_H1']*params['sigma_x2_H1']],
               [params['rho_x1x2_H1']*params['sigma_x1_H1']*params['sigma_x2_H1'], params['sigma_x2_H1']**2]]
        data_2d = np.random.multivariate_normal(mean, cov, N)
        x1 = data_2d[:, 0]
        x2 = data_2d[:, 1]
        # Generate x3 with sinusoidal modulation
        x3 = []
        k = 2*np.pi/params['lambda_x3_H1']
        while len(x3) < N:
            candidate = np.random.uniform(-np.pi, np.pi)
            u = np.random.uniform(0, 1)
            if u < (1 + params['amp_x3_H1'] * np.sin(k * candidate)) / 2:
                x3.append(candidate)
        x3 = np.array(x3)
    else:
        raise ValueError("Hypothesis must be 'H0' or 'H1'")
    return pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})