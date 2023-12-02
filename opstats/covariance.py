from typing import Union, List

from opstats.moments import Moments, MomentCalculator, aggregate_moments


class Covariance:
    """
    Results of covariance calculations.

    Attributes
    ----------
    sample_count int
        the total number of data points
    moments_x: Moments
        the moments for the first series
    moments_y: Moments
        the moments for the second series
    comoment: float
        the calculated co-moment, used for aggregation
    covariance: float
        the calculated population or sample covariance
    correlation: float
        the calculated correlation coefficient
    """

    def __init__(self, moments_x: Moments, moments_y: Moments, comoment: float, covariance: float, correlation: float) -> None:
        assert (moments_x.sample_count == moments_y.sample_count)

        self.sample_count = moments_x.sample_count
        self.moments_x = moments_x
        self.moments_y = moments_y
        self.comoment = comoment
        self.covariance = covariance
        self.correlation = correlation


class CovarianceCalculator:
    """
    Online algorithm for calculating covariance and correlation.
    """

    def __init__(self, sample_covariance: bool = False) -> None:
        """
        Initialise a new calculator.

        Parameters
        ----------
        sample_covariance: bool, optional
            set to True to calculate the sample covariance instead of the population covariance
        """

        if sample_covariance is None:
            raise ValueError('Argument "sample_covariance" must be a bool, received None.')
        elif type(sample_covariance) is not bool:
            raise ValueError(f'Argument "sample_covariance" must be a bool, received {type(sample_covariance)}')

        self._sample_covar = sample_covariance

        self._stats_x = MomentCalculator(sample_variance=sample_covariance)
        self._stats_y = MomentCalculator(sample_variance=sample_covariance)
        self._C = 0

    def add(self, x: Union[int, float], y: Union[int, float]) -> None:
        """
        Adds a new data point.

        Parameters
        ----------
        x:  Union[int, float]
            the first value of the data point to add
        y:  Union[int, float]
            the second value of the data point to add
        """

        if x is not None and y is not None:
            dx = x - self._stats_x._mean
            self._stats_x.add(x)
            self._stats_y.add(y)
            self._C += dx * (y - self._stats_y._mean)

    def get(self) -> Covariance:
        """
        Gets the covariance statistics for all data points added so far.

        Returns
        -------
        CovarianceStats
            the calculated covariance statistics
        """

        x = self._stats_x.get()
        y = self._stats_y.get()
        n = x.sample_count

        assert (n == y.sample_count)

        if n < 1:
            return Covariance(Moments(), Moments(), 0.0, 0.0, 0.0)
        elif self._sample_covar:
            # Bessel's correction for sample variance
            c = 1
        else:
            c = 0

        cov = self._C / (n - c)

        # If all the inputs are the same, standard_deviation will be 0, resulting in a division by 0.
        if x.standard_deviation == 0 or y.standard_deviation == 0:
            cor = 0.0
        else:
            cor = self._C / (x.standard_deviation * y.standard_deviation) / (n - c)

        return Covariance(x, y, self._C, cov, cor)


def aggregate_covariance(stats: List[Covariance], sample_covariance: bool = False) -> Covariance:
    """
    Combines a list of covariance statistics previously calculated in parallel.

    Parameters
    ----------
    stats: List[CovarianceStats]
        list of separate instances of calculated covariances from one data set
    sample_covariance: bool, optional
        population covariance is calculated by default. Set to True to calculate the sample covariance

    Returns
    -------
    CovarianceStats
        the combined covariance statistics
    """

    def _merge(left: Covariance, right: Covariance) -> Covariance:
        x_agg = aggregate_moments([left.moments_x, right.moments_x], sample_variance=sample_covariance)
        y_agg = aggregate_moments([left.moments_y, right.moments_y], sample_variance=sample_covariance)
        com = left.comoment + right.comoment + (left.moments_x.mean - right.moments_x.mean) * (left.moments_y.mean - right.moments_y.mean) * ((left.sample_count * right.sample_count) / (left.sample_count + right.sample_count))
        if sample_covariance:
            # Bessel's correction for sample variance
            c = 1
        else:
            c = 0

        cov = com / (left.sample_count + right.sample_count - c)

        # If all the inputs are the same, standard_deviation will be 0, resulting in a division by 0.
        if x_agg.standard_deviation == 0 or y_agg.standard_deviation == 0:
            cor = 0.0
        else:
            cor = com / (x_agg.standard_deviation * y_agg.standard_deviation) / (x_agg.sample_count - c)

        return Covariance(x_agg, y_agg, com, cov, cor)

    if sample_covariance is None:
        raise ValueError('Argument "sample_covariance" must be a bool, received None.')
    elif type(sample_covariance) is not bool:
        raise ValueError(f'Argument "sample_covariance" must be a bool, received {type(sample_covariance)}')

    if stats is None:
        raise ValueError('Argument "stats" must be a list of CovarianceStats, received None.')
    elif type(stats) is not list:
        raise ValueError(f'Argument "stats" must be a list of CovarianceStats, received {type(stats)}')
    else:
        stats = list(filter(lambda s: s is not None and type(s) is Covariance and s.sample_count > 0, stats))
        if len(stats) == 0:
            return Covariance(Moments(), Moments(), 0.0, 0.0, 0.0)
        elif len(stats) == 1:
            return stats[0]

    result = stats[0]
    for i in range(1, len(stats)):
        result = _merge(result, stats[i])
    return result
