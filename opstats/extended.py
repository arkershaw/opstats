from typing import NamedTuple, List, Union, Tuple, Dict, Optional, Iterable

from opstats.moments import MomentCalculator, aggregate_moments, Moments
from tdigest import TDigest
from hyperloglog import HyperLogLog

__all__ = ['ExtendedStats', 'ParallelStats', 'ExtendedCalculator', 'aggregate_extended']

DEFAULT_ACCURACY = 0.01


class ExtendedStats(NamedTuple):
    """
    Results of extended stats calculations.

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
    cardinality: int
        the approximate number of unique values
    median: float
        the approximate median value
    interquartile_range: float
        the approximate interquartile range
    percentiles: Dict[int, float]
        the calculated percentile values
    """
    sample_count: int = 0
    mean: float = 0.0
    variance: float = 0.0
    standard_deviation: float = 0.0
    skewness: float = 0.0
    kurtosis: float = -3.0
    cardinality: int = 0
    median: float = 0.0
    interquartile_range: float = 0.0
    percentiles: Dict[int, float] = {}


class ParallelStats(NamedTuple):
    """
    Intermediate results of parallel calculations.
    Can be combined with `aggregate_stats(...)` or converted with `.calculate()`.

    Attributes
    ----------
    moments: Moments
        the results of moment calculations
    centroids: List[Tuple[float, int]]
        the list of centroids used for approximating percentiles
    state: Dict[str, str]
        the state required for calculating cardinality
    """
    moments: Moments
    centroids: List[Tuple[float, int]]
    state: Dict[str, str]

    def calculate(self, percentiles: Optional[Iterable[int]] = None) -> ExtendedStats:
        """
        Calculates the moments, cardinality and percentiles and returns the results.

        Parameters
        ----------
        percentiles: Optional[Iterable[int]]
            list of additional percentiles to calculate (mean and interquartile range are always calculated)

        Returns
        -------
        Stats
            the calculated moments, cardinality and percentiles
        """
        if self.moments.sample_count == 0:
            return ExtendedStats()
        else:
            tdigest = TDigest()
            _from_centroids(tdigest, self.centroids)

            hll = HyperLogLog(DEFAULT_ACCURACY)
            hll.__setstate__(self.state)

            median = tdigest.percentile(50)
            perc_25 = tdigest.percentile(25)
            perc_75 = tdigest.percentile(75)
            iq_r = perc_75 - perc_25

            pc_res: Dict[int, float] = {}
            if percentiles is None:
                pc_res = {}
            else:
                for pc in percentiles:
                    if pc == 50:
                        pc_res[pc] = median
                    elif pc == 25:
                        pc_res[pc] = perc_25
                    elif pc == 75:
                        pc_res[pc] = perc_75
                    else:
                        pc_res[pc] = tdigest.percentile(pc)

            return ExtendedStats(
                self.moments.sample_count,
                self.moments.mean,
                self.moments.variance,
                self.moments.standard_deviation,
                self.moments.skewness,
                self.moments.kurtosis,
                round(hll.card()),
                median,
                iq_r,
                pc_res
            )


def _get_centroids(tdigest: TDigest) -> List[Tuple[float, int]]:
    return [(float(c['m']), int(c['c'])) for c in tdigest.centroids_to_list()]


def _from_centroids(tdigest: TDigest, centroids: List[Tuple[float, int]]) -> None:
    c = [{'m': c[0], 'c': c[1]} for c in centroids]
    tdigest.update_centroids_from_list(c)


class ExtendedCalculator:
    """
    Online algorithm for calculating:
      Mean
      Variance
      Skewness
      Kurtosis
      Cardinality
      Median
      Interquartile Range
    """

    def __init__(self, sample_variance: bool = False, bias_adjust: bool = False, error_rate: float = DEFAULT_ACCURACY) -> None:
        """
        Initialise a new calculator.

        Parameters
        ----------
        sample_variance: bool, optional
            set to True to calculate the sample varaiance instead of the population variance
        bias_adjust: bool, optional
            set to True to adjust skewness and kurtosis for bias (adjusted Fisher-Pearson)
        error_rate: float
            the accuracy of the cardinality estimation
        """
        self._moment_calc = MomentCalculator(sample_variance, bias_adjust)
        self._tdigest = TDigest()
        self._hll = HyperLogLog(error_rate)

    def add(self, x: Union[int, float]) -> None:
        """
        Adds a new data point.

        Parameters
        ----------
        x:  Union[int, float]
            the data point to add
        """

        if x is not None:
            self._moment_calc.add(x)
            self._tdigest.update(x)
            self._hll.add(str(x))

    def get_parallel(self) -> ParallelStats:
        """
        Gets the intermediate stats for all data points added so far.
        Used when calculating stats in parallel.
        Can be combined with `aggregate_stats(...)` or converted with `.calculate()`.

        Returns
        -------
        Stats
            named tuple containing the calculated stats
        """
        moments = self._moment_calc.get()
        centroids = _get_centroids(self._tdigest)
        return ParallelStats(moments, centroids, self._hll.__getstate__())

    def get(self) -> ExtendedStats:
        """
        Gets the stats for all data points added so far.

        Returns
        -------
        Stats
            named tuple containing the calculated stats
        """
        return self.get_parallel().calculate()


def aggregate_extended(stats: List[ParallelStats], sample_variance: bool = False, bias_adjust: bool = False) -> ParallelStats:
    """
    Combines a list of ParallelStats values previously calculated in parallel.

    Parameters
    ----------
    stats: List[ParallelStats]
        list of separate instances of calculated ParallelStats from one data set
    sample_variance: bool, optional
        population variance is calculated by default. Set to True to calculate the sample varaiance
    bias_adjust: bool, optional
        set to True to adjust skewness and kurtosis for bias (adjusted Fisher-Pearson)

    Returns
    -------
    ParallelStats
        the combined values
    """
    if sample_variance is None:
        raise ValueError('Argument "sample_variance" must be a bool, received None.')
    elif type(sample_variance) is not bool:
        raise ValueError(f'Argument "sample_variance" must be a bool, received {type(sample_variance)}')

    if bias_adjust is None:
        raise ValueError('Argument "bias_adjust" must be a bool, received None.')
    elif type(bias_adjust) is not bool:
        raise ValueError(f'Argument "bias_adjust" must be a bool, received {type(bias_adjust)}')

    if stats is None:
        raise ValueError('Argument "stats" must be a list of ParallelStats, received None.')
    elif type(stats) is not list:
        raise ValueError(f'Argument "stats" must be a list of ParallelStats, received {type(stats)}')
    else:
        stats = list(filter(lambda s: s is not None and type(s) is ParallelStats and s.moments.sample_count > 0, stats))
        if len(stats) == 0:
            return ParallelStats(Moments(), [], {})
        elif len(stats) == 1:
            return stats[0]

    hll: Optional[HyperLogLog] = None
    tdigest = TDigest()
    moments: List[Moments] = []

    for s in stats:
        if hll is None:
            hll = HyperLogLog(DEFAULT_ACCURACY)
            hll.__setstate__(s.state)
        else:
            hll_new = HyperLogLog(DEFAULT_ACCURACY)
            hll_new.__setstate__(s.state)
            hll.update(hll_new)

        _from_centroids(tdigest, s.centroids)
        moments.append(s.moments)

    if hll is None:
        # Shouldn't happen but required for linting.
        state = {}
    else:
        state = hll.__getstate__()

    return ParallelStats(
        aggregate_moments(moments, sample_variance, bias_adjust),
        _get_centroids(tdigest),
        state
    )
