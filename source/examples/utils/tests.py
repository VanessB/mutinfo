import numpy
from tqdm import tqdm
from scipy.stats._multivariate import multi_rv_frozen

from collections.abc import Callable, Iterable


def run_tests(distribution_factory: Callable[[float], multi_rv_frozen],
              estimator: MutualInformationEstimator,
              MI_grid: Iterable[float], n_samples: int, n_runs: int) -> numpy.ndarray:
    """
    Iteratively run mutual information estimation tests.

    Parameters
    ----------
    distribution_factory : Callable[[float], multi_rv_frozen]
        A factory getting a ground truth value of MI and yielding
        a corresponding distribution.
    estimator : MutualInformationEstimator
        Mutual information estimator.
    MI_grid : Iterable[float]
        Ground truth values of the mutual information,
        used to perform the tests.
    n_samples : int
        Number of samples to generate during each test.
    n_runs : int
        Number of runs used to average the results.

    Returns
    -------
    estimated_MI : numpy.ndarray
        Estimated values of the mutual information
        and corresponding standard deviations.
    """
    
    estimated_MI = []

    for mutual_information in tqdm(MI_grid):
        current_run_estimates = []
        for run in range(n_runs):
            random_variable = distribution_factory(mutual_information)
            x, y = random_variable.rvs(n_samples)
        
            current_run_estimates.append(estimator(x, y))
        
        current_run_estimates = numpy.array(current_run_estimates)
        mean = numpy.mean(current_run_estimates)
        std = numpy.std(current_run_estimates) / numpy.sqrt(n_runs)
        
        estimated_MI.append([mean, std])
    
    estimated_MI = numpy.array(estimated_MI)
    
    return estimated_MI