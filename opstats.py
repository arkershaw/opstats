from math import sqrt
from typing import NamedTuple, List, Union
from dataclasses import dataclass


class InvalidStateException(Exception):
    pass


class Stats(NamedTuple):
    """
    Results of moment calculations.

    Attributes
    ----------
    count: int
        the total number of data points
    mean: float
        the mean value of all data points
    variance: float
        the calculated variance (population or sample depending on parameters)
    standard_deviation: float
        the standard deviation (sqrt(variance)) for convenience
    skew: float
        the skewness (Fisher-Pearson)
    kurtosis: float
        the kurtosis (Pearson)
    """

    sample_count: int
    mean: float
    variance: float
    standard_deviation: float
    skewness: float
    kurtosis: float


# Adapted from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
class OnlineCalculator:
    """
    Online algorithm for calculating mean, variance, skewness and kurtosis.
    """

    def __init__(self, sample_variance: bool = False) -> None:
        """
        Initialise a new calculator.

        Parameters
        ----------
        sample_variance: bool, optional
            set to True to calculate the sample varaiance instead of the population variance
        """

        self._sample_var = sample_variance
        self._n = 0
        self._mean = 0
        self._M2 = 0
        self._M3 = 0
        self._M4 = 0

    def add(self, x: Union[int, float]) -> None:
        """
        Adds a new data point.

        Parameters
        ----------
        x:  Union[int, float]
            the data point to add
        """

        n1 = self._n
        self._n = self._n + 1
        delta = x - self._mean
        delta_n = delta / self._n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n1
        self._mean = self._mean + delta_n
        self._M4 = self._M4 + term1 * delta_n2 * (self._n * self._n - 3 * self._n + 3) + 6 * delta_n2 * self._M2 - 4 * delta_n * self._M3
        self._M3 = self._M3 + term1 * delta_n * (self._n - 2) - 3 * delta_n * self._M2
        self._M2 = self._M2 + term1

    def get(self) -> Stats:
        """
        Gets the statistics for all data points added so far.

        Returns
        -------
        Stats
            named tuple containing the calculated statistics

        Raises
        ------
        InvalidStateException
            less than two data points have been added
        """

        if self._n < 2:
            raise InvalidStateException('At least two data points must be added.')

        if self._sample_var:
            var = self._M2 / (self._n - 1)
        else:
            var = self._M2 / self._n

        # If all the inputs are the same, M2 will be 0, resulting in a division by 0.
        if self._M2 > 0:
            skew = (sqrt(self._n) * self._M3) / (self._M2 ** (3 / 2))
            kurt = (self._n * self._M4) / (self._M2 * self._M2) - 3
        else:
            skew = 0.0
            kurt = -3.0

        return Stats(self._n, self._mean, var, sqrt(var), skew, kurt)


# Translated from https://rdrr.io/cran/utilities/src/R/sample.decomp.R
def aggregate_stats(stats: List[Stats], sample_variance: bool = False) -> Stats:
    """
    Combines a list of Stats tuples previously calculated in parallel.

    Parameters
    ----------
    stats: List[Stats]
        list of separate instances of calculated statistics from one data set
    sample_variance: bool, optional
        population variance is calculated by default. Set to True to calculate the sample varaiance

    Returns
    -------
    Stats
        the combined statistics
    """

    @dataclass
    class Pool:
        n: int = 0
        mean: float = 0.0
        SS: float = 0.0
        var: float = 0.0
        sd: float = 0.0
        SC: float = 0.0
        skew: float = 0.0
        SQ: float = 0.0
        kurt: float = 0.0

    pool = Pool()

    # First pass - calculate the mean.
    SS = []
    sum_mean = 0.0
    sum_ss = 0.0
    for sample in stats:
        pool.n += sample.sample_count
        sum_mean += (sample.mean * sample.sample_count)
        if sample_variance:
            _ss = (sample.sample_count - 1) * sample.variance
        else:
            _ss = sample.sample_count * sample.variance
        SS.append(_ss)
        sum_ss += _ss

    pool.mean = sum_mean / pool.n

    # Second pass - calculate the variance and standard deviation.
    deviation = []
    sum_n_dev_2 = 0.0
    sum_n_dev_3 = 0.0
    sum_ss_dev = 0.0
    sum_ss_dev_2 = 0.0
    sum_n_dev_4 = 0.0
    for i in range(len(stats)):
        sample = stats[i]
        _dev = sample.mean - pool.mean
        deviation.append(_dev)
        sum_n_dev_2 += sample.sample_count * _dev ** 2
        sum_ss_dev += SS[i] * _dev
        sum_n_dev_3 += sample.sample_count * _dev ** 3
        sum_ss_dev_2 += SS[i] * _dev ** 2
        sum_n_dev_4 += sample.sample_count * _dev ** 4

    pool.SS = sum_ss + sum_n_dev_2
    if sample_variance:
        pool.var = pool.SS / (pool.n - 1)
    else:
        pool.var = pool.SS / pool.n
    pool.sd = sqrt(pool.var)

    # Third pass - calculate the skew and kurtosis.
    # TODO: Implement Adjusted Fisher Pearson.
    def skew_adj(n: float) -> float:
        return 1
        return ((n - 1) / n) ** (3 / 2)  # 'Moment', 'b', 'Minitab'
        return sqrt(n * (n - 1)) / (n - 2)  # 'Adjusted Fisher Pearson', 'G', 'Excel', 'SPSS', 'SAS'

    def kurt_adj(n: float) -> float:
        return 1
        return ((n - 1) / n) ** 2  # 'Moment', 'b', 'Minitab'
        return (n + 1) * (n - 1) / ((n - 2) * (n - 3))  # 'Adjusted Fisher Pearson', 'G', 'Excel', 'SPSS', 'SAS'

    def excess_adj(n: float) -> float:
        return -3  # Excess
        return -3 * (n - 1) ** 2 / ((n - 2) * (n - 3))  # Excess 'Adjusted Fisher Pearson', 'G', 'Excel', 'SPSS', 'SAS

    SC = []
    SQ = []
    sum_sc = 0.0
    sum_sq = 0.0
    sum_sc_dev = 0.0
    for i in range(len(stats)):
        sample = stats[i]
        _sc = sample.skewness * (SS[i] ** (3 / 2)) / (skew_adj(sample.sample_count) * sqrt(sample.sample_count))
        SC.append(_sc)
        sum_sc += _sc
        _sq = (sample.kurtosis - excess_adj(sample.sample_count)) * SS[i] ** 2 / (kurt_adj(sample.sample_count) * sample.sample_count)
        SQ.append(_sq)
        sum_sq += _sq
        sum_sc_dev += _sc * deviation[i]

    pool.SC = sum_sc + 3 * sum_ss_dev + sum_n_dev_3
    pool.skew = skew_adj(pool.n) * sqrt(pool.n) * pool.SC / pool.SS ** (3 / 2)
    pool.SQ = sum_sq + 4 * sum_sc_dev + 6 * sum_ss_dev_2 + sum_n_dev_4
    pool.kurt = kurt_adj(pool.n) * pool.n * pool.SQ / pool.SS ** 2 + excess_adj(pool.n)

    return Stats(
        pool.n,
        pool.mean,
        pool.var,
        pool.sd,
        pool.skew,
        pool.kurt
    )
