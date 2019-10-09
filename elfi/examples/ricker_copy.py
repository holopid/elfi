"""Example implementation of the Ricker model."""

from functools import partial
from itertools import combinations
from sklearn.linear_model import LinearRegression

import numpy as np
import scipy.stats as ss

import elfi
import logging

import warnings
warnings.filterwarnings('ignore')  # ignore polyfit rank warnings


def ricker(log_rate, stock_init=1., n_obs=50, batch_size=1, random_state=None):
    """Generate samples from the Ricker model.
    Ricker, W. E. (1954) Stock and Recruitment Journal of the Fisheries
    Research Board of Canada, 11(5): 559-623.
    Parameters
    ----------
    log_rate : float or np.array
        Log growth rate of population.
    stock_init : float or np.array, optional
        Initial stock.
    n_obs : int, optional
    batch_size : int, optional
    random_state : np.random.RandomState, optional
    Returns
    -------
    stock : np.array
    """
    random_state = random_state or np.random

    stock = np.empty((batch_size, n_obs))
    stock[:, 0] = stock_init

    for ii in range(1, n_obs):
        stock[:, ii] = stock[:, ii - 1] * np.exp(log_rate - stock[:, ii - 1])

    return stock


def stochastic_ricker(log_rate,
                      std,
                      scale,
                      stock_init=1.,
                      n_obs=50,
                      batch_size=1,
                      random_state=None):
    """Generate samples from the stochastic Ricker model.
    Here the observed stock ~ Poisson(true stock * scaling).
    Parameters
    ----------
    log_rate : float or np.array
        Log growth rate of population.
    std : float or np.array
        Standard deviation of innovations.
    scale : float or np.array
        Scaling of the expected value from Poisson distribution.
    stock_init : float or np.array, optional
        Initial stock.
    n_obs : int, optional
    batch_size : int, optional
    random_state : np.random.RandomState, optional
    Returns
    -------
    stock_obs : np.array
    """
    random_state = random_state or np.random

    stock_obs = np.empty((batch_size, n_obs))
    stock_prev = stock_init

    for ii in range(n_obs):
        stock = stock_prev * np.exp(log_rate - stock_prev + std * random_state.randn(batch_size))
        stock_prev = stock

        # the observed stock is Poisson distributed
        stock_obs[:, ii] = random_state.poisson(scale * stock, batch_size)

    return stock_obs


def get_model(n_obs=50, true_params=None, bounds=None, seed_obs=None, stochastic=True, n_lags=5):
    """Return a complete Ricker model in inference task.
    This is a simplified example that achieves reasonable predictions. For more extensive treatment
    and description using 13 summary statistics, see:
    Wood, S. N. (2010) Statistical inference for noisy nonlinear ecological dynamic systems,
    Nature 466, 1102–1107.
    Parameters
    ----------
    n_obs : int, optional
        Number of observations.
    true_params : list, optional
        Parameters with which the observed data is generated.
    seed_obs : int, optional
        Seed for the observed data generation.
    stochastic : bool, optional
        Whether to use the stochastic or deterministic Ricker model.
    Returns
    -------
    m : elfi.ElfiModel
    """
    logger = logging.getLogger()
    if stochastic:
        simulator = partial(stochastic_ricker, n_obs=n_obs)
        if true_params is None:
            true_params = [3.8, 0.3, 10.]

    else:
        simulator = partial(ricker, n_obs=n_obs)
        if true_params is None:
            true_params = [3.8]

    m = elfi.ElfiModel()
    y_obs = simulator(*true_params, n_obs=n_obs, random_state=np.random.RandomState(seed_obs))
    sim_fn = partial(simulator, n_obs=n_obs)
    sumstats = []

    if stochastic:
        if bounds is not None:
            # Fix the prior distributions
            elfi.Prior('uniform', bounds['t1'][0], bounds['t1'][1] - bounds['t1'][0], model=m, name='t1')
            elfi.Prior('uniform', bounds['t2'][0], bounds['t2'][1] - bounds['t2'][0], model=m, name='t2')
            elfi.Prior('uniform', bounds['t3'][0], bounds['t3'][1] - bounds['t3'][0], model=m, name='t3')
        else:
            elfi.Prior('uniform', 3, 2, model=m, name='t1')
            elfi.Prior('uniform', 0, 0.6, model=m, name='t2')
            elfi.Prior('uniform', 5, 10, model=m, name='t3')
        elfi.Simulator(sim_fn, m['t1'], m['t2'], m['t3'], observed=y_obs, name='Ricker')
        sumstats.append(elfi.Summary(mean, m['Ricker'], name='mu'))
        sumstats.append(elfi.Summary(num_zeros, m['Ricker'], name='zeros'))
        for i in range(1, n_lags + 1):
            ss = elfi.Summary(partial(autocov, lag=i), m['Ricker'], name='autocov_{}'.format(i))
            sumstats.append(ss)
        # for i, j in combinations(range(1, n_lags + 1), 2):
        #     ss = elfi.Summary(partial(pairwise_autocov, lag_i=i, lag_j=j), m['Ricker'], name='pw_autocov_{}_{}'.format(i, j))
        #     sumstats.append(ss)
        sumstats.append(elfi.Summary(least_square_estimates, m['Ricker'], name='lse'))
        sumstats.append(elfi.Summary(partial(cubic_regression, x_obs=y_obs), m['Ricker'], name='cubic'))
        # elfi.Discrepancy(chi_squared, *sumstats, name='d')
        elfi.Distance('euclidean', *sumstats, name='d')
        elfi.Operation(np.log, m['d'], name='log_d')

    else:  # very simple deterministic case
        elfi.Prior(ss.expon, np.e, model=m, name='t1')
        elfi.Simulator(sim_fn, m['t1'], observed=y_obs, name='Ricker')
        sumstats.append(elfi.Summary(partial(np.mean, axis=1), m['Ricker'], name='Mean'))
        elfi.Distance('euclidean', *sumstats, name='d')

    logger.info("Generated observations with true parameters "
                "log_rate: %.1f, std: %.1f, scale: %.1f ", *true_params)

    return m


def chi_squared(*simulated, observed):
    """Return Chi squared goodness of fit.
    Adjusts for differences in magnitude between dimensions.
    Parameters
    ----------
    simulated : np.arrays
    observed : tuple of np.arrays
    """
    simulated = np.column_stack(simulated)
    observed = np.column_stack(observed)
    d = np.sum((simulated - observed)**2. / observed, axis=1)
    return d


def num_zeros(x):
    """Return a summary statistic: number of zero observations."""
    n = np.sum(x == 0, axis=1)
    return n


def autocov(x, lag=1):
    """Return a summary statistic: autocovariance."""
    n_obs = x.shape[1]
    mu_x = np.mean(x, axis=1)
    std_x = np.std(x, axis=1, ddof=1)
    sx = ((x.T - mu_x) / std_x).T
    sx_t = sx[:, lag:]
    sx_s = sx[:, :-lag]
    c = np.sum(sx_t * sx_s, axis=1)
    return c


def pairwise_autocov(x, lag_i=1, lag_j=1):
    """Return a summary statistic: pairwise autocovariance."""
    ac_i = autocov(x, lag=lag_i)
    ac_j = autocov(x, lag=lag_j)
    return ac_i * ac_j


def cubic_regression(x, x_obs):
    """Return a summary statisic: cubic regression estimates."""
    x_diff = np.diff(x, 1)
    x_obs_diff = np.diff(x_obs, 1)
    res = [np.polyfit(x_diff[i, :], x_obs_diff[0, :], 3) for i in range(x_diff.shape[0])]
    return np.array(res)


def least_square_estimates(x):
    """Return a summary statistic: least square estimates."""
    c = np.apply_along_axis(_lse, 1, x)
    return c


def _lse(x):
    """Helpper function for least square estimates."""
    X = np.array([x[:-1]**0.3, x[:-1]**0.6]).T
    y = np.array(x[1:]**0.3)
    m = LinearRegression(fit_intercept=False)
    m.fit(X, y)
    return m.coef_


def mean(x):
    """Return a summary statistic: mean."""
    return np.mean(x, axis=1)
