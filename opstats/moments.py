from math import sqrt
from typing import NamedTuple, List, Union

__all__ = ['Moments', 'MomentCalculator', 'aggregate_moments']


class Moments(NamedTuple):
    """
    Results of moment calculations.

    Attributes
    ----------
    sample_count: int
        the total number of data points
    mean: float
        the mean value of all data points
    variance: float
        the calculated population or sample variance
    standard_deviation: float
        the standard deviation (sqrt(variance)) for convenience
    skew: float
        the skewness
    kurtosis: float
        the excess kurtosis
    """

    sample_count: int = 0
    mean: float = 0.0
    variance: float = 0.0
    standard_deviation: float = 0.0
    skewness: float = 0.0
    kurtosis: float = -3.0


# Adapted from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
class MomentCalculator:
    """
    Online algorithm for calculating mean, variance, skewness and kurtosis.
    """

    def __init__(self, sample_variance: bool = False, bias_adjust: bool = False) -> None:
        """
        Initialise a new calculator.

        Parameters
        ----------
        sample_variance: bool, optional
            set to True to calculate the sample varaiance instead of the population variance
        bias_adjust: bool, optional
            set to True to adjust skewness and kurtosis for bias (adjusted Fisher-Pearson)
        """

        if sample_variance is None:
            raise ValueError('Argument "sample_variance" must be a bool, received None.')
        elif type(sample_variance) is not bool:
            raise ValueError(f'Argument "sample_variance" must be a bool, received {type(sample_variance)}')

        if bias_adjust is None:
            raise ValueError('Argument "bias_adjust" must be a bool, received None.')
        elif type(bias_adjust) is not bool:
            raise ValueError(f'Argument "bias_adjust" must be a bool, received {type(bias_adjust)}')

        self._sample_var = sample_variance
        self._adjust = bias_adjust

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

        if x is not None:
            n1 = self._n
            self._n += 1
            delta = x - self._mean
            delta_n = delta / self._n
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * n1
            self._mean += delta_n
            self._M4 += term1 * delta_n2 * (self._n * self._n - 3 * self._n + 3) + 6 * delta_n2 * self._M2 - 4 * delta_n * self._M3
            self._M3 += term1 * delta_n * (self._n - 2) - 3 * delta_n * self._M2
            self._M2 += term1

    def get(self) -> Moments:
        """
        Gets the moments for all data points added so far.

        Returns
        -------
        Moments
            named tuple containing the calculated moments
        """

        if self._n < 1:
            return Moments()
        elif self._n == 1:
            # There will be no variance for a single data point.
            return Moments(self._n, self._mean)

        if self._sample_var:
            var = self._M2 / (self._n - 1)
        else:
            var = self._M2 / self._n

        # If all the inputs are the same, M2 will be 0, resulting in a division by 0.
        if self._M2 > 0:
            skew = (sqrt(self._n) * self._M3) / (self._M2 ** (3 / 2))
            kurt = (self._n * self._M4) / (self._M2 * self._M2)
            if self._adjust:
                skew = skew * sqrt(self._n * (self._n - 1)) / (self._n - 2)
                kurt = kurt * (self._n + 1) * (self._n - 1) / ((self._n - 2) * (self._n - 3)) - 3 * (self._n - 1) ** 2 / ((self._n - 2) * (self._n - 3))
            else:
                kurt = kurt - 3
            return Moments(self._n, self._mean, var, sqrt(var), skew, kurt)
        else:
            return Moments(self._n, self._mean, var)


# Translated from https://rdrr.io/cran/utilities/src/R/sample.decomp.R
def aggregate_moments(moments: List[Moments], sample_variance: bool = False, bias_adjust: bool = False) -> Moments:
    """
    Combines a list of Moment tuples previously calculated in parallel.

    Parameters
    ----------
    moments: List[Moments]
        list of separate instances of calculated moments from one data set
    sample_variance: bool, optional
        population variance is calculated by default. Set to True to calculate the sample varaiance
    bias_adjust: bool, optional
        set to True to adjust skewness and kurtosis for bias (adjusted Fisher-Pearson)

    Returns
    -------
    Moments
        the combined moments
    """

    if sample_variance is None:
        raise ValueError('Argument "sample_variance" must be a bool, received None.')
    elif type(sample_variance) is not bool:
        raise ValueError(f'Argument "sample_variance" must be a bool, received {type(sample_variance)}')

    if bias_adjust is None:
        raise ValueError('Argument "bias_adjust" must be a bool, received None.')
    elif type(bias_adjust) is not bool:
        raise ValueError(f'Argument "bias_adjust" must be a bool, received {type(bias_adjust)}')

    if moments is None:
        raise ValueError('Argument "moments" must be a list of Moments, received None.')
    elif type(moments) is not list:
        raise ValueError(f'Argument "moments" must be a list of Moments, received {type(moments)}')
    else:
        moments = list(filter(lambda s: s is not None and type(s) is Moments and s.sample_count > 0, moments))
        if len(moments) == 0:
            return Moments()
        elif len(moments) == 1:
            return moments[0]

    class Pool:
        def __init__(self) -> None:
            self.n: int = 0
            self.mean: float = 0.0
            self.SS: float = 0.0
            self.var: float = 0.0
            self.sd: float = 0.0
            self.SC: float = 0.0
            self.skew: float = 0.0
            self.SQ: float = 0.0
            self.kurt: float = 0.0

    pool = Pool()

    # First pass - calculate the mean.
    SS = []
    sum_mean = 0.0
    sum_ss = 0.0
    for sample in moments:
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
    for i in range(len(moments)):
        sample = moments[i]
        _dev = sample.mean - pool.mean
        deviation.append(_dev)
        sum_n_dev_2 += sample.sample_count * _dev ** 2
        sum_ss_dev += SS[i] * _dev
        sum_n_dev_3 += sample.sample_count * _dev ** 3
        sum_ss_dev_2 += SS[i] * _dev ** 2
        sum_n_dev_4 += sample.sample_count * _dev ** 4

    pool.SS = sum_ss + sum_n_dev_2

    # If all the inputs are the same, SS will be 0, resulting in a division by 0.
    if pool.SS == 0.0:
        return Moments(pool.n, pool.mean)

    if sample_variance:
        pool.var = pool.SS / (pool.n - 1)
    else:
        pool.var = pool.SS / pool.n
    pool.sd = sqrt(pool.var)

    # Third pass - calculate the skew and kurtosis.
    def skew_adj(n: float) -> float:
        if bias_adjust:
            return sqrt(n * (n - 1)) / (n - 2)
        else:
            return 1

    def kurt_adj(n: float) -> float:
        if bias_adjust:
            return (n + 1) * (n - 1) / ((n - 2) * (n - 3))
        else:
            return 1

    def excess_adj(n: float) -> float:
        if bias_adjust:
            return -3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        else:
            return -3

    SC = []
    SQ = []
    sum_sc = 0.0
    sum_sq = 0.0
    sum_sc_dev = 0.0
    for i in range(len(moments)):
        sample = moments[i]
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

    return Moments(pool.n, pool.mean, pool.var, pool.sd, pool.skew, pool.kurt)
