from typing import NamedTuple, List, Union, Tuple, Dict, Optional, Iterable
import math

from tdigest import TDigest
from hyperloglog import HyperLogLog

from opstats.moments import MomentCalculator, aggregate_moments, Moments
from opstats.utils import to_numeric, percentile

__all__ = ['ExtendedStats', 'ParallelStats', 'ExtendedCalculator', 'aggregate_extended']

DEFAULT_ACCURACY = 0.01
Centroids = List[Tuple[float, int]]


class HLLState(NamedTuple):
    """
    Wrapper class for HyperLogLog state.
    """
    alpha: float
    p: int
    m: int
    M: List[int]


def _get_centroids(tdigest: TDigest) -> Centroids:
    return [(float(c['m']), int(c['c'])) for c in tdigest.centroids_to_list()]


def _from_centroids(tdigest: TDigest, centroids: Centroids) -> None:
    c = [{'m': c[0], 'c': c[1]} for c in centroids]
    tdigest.update_centroids_from_list(c)


def _get_state(hll: Optional[HyperLogLog] = None) -> HLLState:
    if hll is None:
        hll = HyperLogLog(DEFAULT_ACCURACY)

    assert hll.__slots__ == ('alpha', 'p', 'm', 'M')

    state = hll.__getstate__()

    return HLLState(state['alpha'], state['p'], state['m'], state['M'])


def _from_state(hll: HyperLogLog, state: HLLState) -> HyperLogLog:
    assert hll.__slots__ == ('alpha', 'p', 'm', 'M')

    state = {
        'alpha': state.alpha,
        'p': state.p,
        'm': state.m,
        'M': state.M,
    }

    hll.__setstate__(state)


class ExtendedStats(NamedTuple):
    """
    Results of extended stats calculations.

    Attributes
    ----------
    sample_count: int
        The total number of data points
    mean: float
        The mean value of all data points
    variance: float
        The calculated population or sample variance
    standard_deviation: float
        The standard deviation (sqrt(variance)) for convenience
    skew: float
        The skewness
    kurtosis: float
        The excess kurtosis
    cardinality: int
        The approximate number of unique values
    median: float
        The approximate median value
    interquartile_range: float
        The approximate interquartile range
    percentiles: Dict[int, float]
        The calculated percentile values
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
        The results of moment calculations
    centroids: List[Tuple[float, int]]
        The list of centroids used for approximating percentiles
    state: HLLState
        The state required for calculating cardinality
    values: List[Union[int, float, str]]
        The raw values if estimates are not being used
    """
    moments: Moments = Moments()
    centroids: Centroids = []
    state: HLLState = _get_state()
    values: List[Union[int, float, str]] = []

    def calculate(self, percentiles: Optional[Iterable[int]] = None) -> ExtendedStats:
        """
        Calculates the moments, cardinality and percentiles and returns the results.

        Parameters
        ----------
        percentiles: Optional[Iterable[int]]
            List of additional percentiles to calculate (mean and interquartile range are always calculated)

        Returns
        -------
        Stats
            The calculated moments, cardinality and percentiles
        """
        if self.moments.sample_count == 0:
            return ExtendedStats()
        else:
            pc_res: Dict[int, float] = {}

            if len(self.values) == 0:
                tdigest = TDigest()
                _from_centroids(tdigest, self.centroids)

                hll = HyperLogLog(DEFAULT_ACCURACY)
                _from_state(hll, self.state)
                card = round(hll.card())

                median = tdigest.percentile(50)
                perc_25 = tdigest.percentile(25)
                perc_75 = tdigest.percentile(75)
                iq_r = perc_75 - perc_25

                if percentiles is not None:
                    for pc in percentiles:
                        if pc == 50:
                            pc_res[pc] = median
                        elif pc == 25:
                            pc_res[pc] = perc_25
                        elif pc == 75:
                            pc_res[pc] = perc_75
                        else:
                            pc_res[pc] = tdigest.percentile(pc)
            else:
                card = len(set(self.values))
                sv = sorted([to_numeric(v) for v in self.values])

                median = percentile(sv, 50)
                perc_25 = percentile(sv, 25)
                perc_75 = percentile(sv, 75)
                iq_r = perc_75 - perc_25

                if percentiles is not None:
                    for pc in percentiles:
                        if pc == 50:
                            pc_res[pc] = median
                        elif pc == 25:
                            pc_res[pc] = perc_25
                        elif pc == 75:
                            pc_res[pc] = perc_75
                        else:
                            pc_res[pc] = percentile(sv, pc)

            return ExtendedStats(
                self.moments.sample_count,
                self.moments.mean,
                self.moments.variance,
                self.moments.standard_deviation,
                self.moments.skewness,
                self.moments.kurtosis,
                card,
                median,
                iq_r,
                pc_res
            )


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

    def __init__(self, sample_variance: bool = False, bias_adjust: bool = False,
                error_rate: float = DEFAULT_ACCURACY,
                estimate_threshold: int = -1) -> None:
        """
        Initialise a new calculator.

        Parameters
        ----------
        sample_variance: bool, optional
            Set to True to calculate the sample varaiance instead of the population variance
        bias_adjust: bool, optional
            Set to True to adjust skewness and kurtosis for bias (adjusted Fisher-Pearson)
        error_rate: float
            The accuracy of the cardinality estimation
        estimate_threshold: int
            Use exact cardinality and percentiles until this number of values is reached,
            then switch to HyperLogLog and TDigest algorithms.
            Use 0 to always estimate and -1 to calculate the size based on the error rate.

        """
        self._error_rate = error_rate
        self._moment_calc = MomentCalculator(sample_variance, bias_adjust)
        self._tdigest = None
        self._hll = None
        self._values = []

        if estimate_threshold < 0:
            p = int(math.ceil(math.log((1.04 / error_rate) ** 2, 2)))
            self._estimate_threshold = 1 << p
        else:
            self._estimate_threshold = estimate_threshold
            if estimate_threshold == 0:
                self._tdigest = TDigest()
                self._hll = HyperLogLog(error_rate)
                self._values = None

    def add(self, x: Union[int, float, str]) -> None:
        """
        Adds a new data point.
        Strings will be converted to numeric. For alphanumeric strings, the length will be used.

        Parameters
        ----------
        x:  Union[int, float, str]
            The data point to add
        """

        nx = to_numeric(x)

        if nx is not None:
            self._moment_calc.add(nx)

            if self._values is None:
                self._tdigest.update(nx)
                self._hll.add(str(x))
            else:
                self._values.append(x)
                if len(self._values) >= self._estimate_threshold:
                    self._tdigest = TDigest()
                    self._tdigest.batch_update([to_numeric(v) for v in self._values])
                    self._hll = HyperLogLog(self._error_rate)
                    for x2 in self._values:
                        self._hll.add(str(x2))
                    self._values = None

    def get_parallel(self) -> ParallelStats:
        """
        Gets the intermediate stats for all data points added so far.
        Used when calculating stats in parallel.
        Can be combined with `aggregate_stats(...)` or converted with `.calculate()`.

        Returns
        -------
        Stats
            Named tuple containing the calculated stats
        """
        moments = self._moment_calc.get()
        if self._values is None:
            centroids = _get_centroids(self._tdigest)
            state = _get_state(self._hll)
            values = []
        else:
            centroids = []
            state = _get_state()
            values = self._values
        return ParallelStats(moments, centroids, state, values)

    def get(self) -> ExtendedStats:
        """
        Gets the stats for all data points added so far.

        Returns
        -------
        Stats
            Named tuple containing the calculated stats
        """
        return self.get_parallel().calculate()


def aggregate_extended(stats: List[ParallelStats], sample_variance: bool = False, bias_adjust: bool = False) -> ParallelStats:
    """
    Combines a list of ParallelStats values previously calculated in parallel.

    Parameters
    ----------
    stats: List[ParallelStats]
        List of separate instances of calculated ParallelStats from one data set
    sample_variance: bool, optional
        Population variance is calculated by default. Set to True to calculate the sample varaiance
    bias_adjust: bool, optional
        Set to True to adjust skewness and kurtosis for bias (adjusted Fisher-Pearson)

    Returns
    -------
    ParallelStats
        The combined values
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
    values = []

    for s in stats:
        moments.append(s.moments)

        if len(s.values) == 0:
            # This result is an estimate
            if hll is None:
                # Create new HLL object and populate
                hll = HyperLogLog(DEFAULT_ACCURACY)
                _from_state(hll, s.state)
                # Populate the TDigest centroids
                _from_centroids(tdigest, s.centroids)
                # Add any existing values when we first switch to estimators
                if len(values) > 0:
                    for v in values:
                        hll.add(str(v))
                    tdigest.batch_update([to_numeric(v) for v in values])
                    values = []
            else:
                # Update existing estimators from HLL state and centroids
                hll_new = HyperLogLog(DEFAULT_ACCURACY)
                _from_state(hll_new, s.state)
                hll.update(hll_new)
                # Create local new estimators and combine
                _from_centroids(tdigest, s.centroids)
        elif hll is None:
            # No stats have exceeded the threshold yet
            # Add the raw values to the list
            values += s.values
        else:
            # We are already estimating
            # Add the raw values to the estimators
            for v in s.values:
                hll.add(str(v))
            tdigest.batch_update([to_numeric(v) for v in s.values])

    moments = aggregate_moments(moments, sample_variance, bias_adjust)

    if len(values) > 0:
        return ParallelStats(
            moments,
            [],
            _get_state(),
            values
        )
    else:
        centroids = _get_centroids(tdigest)
        state = _get_state(hll)

        return ParallelStats(
            moments,
            centroids,
            state,
            []
        )
