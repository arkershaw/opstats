import unittest
from typing import List, Union

import numpy
import scipy.stats

from opstats.moments import Moments, MomentCalculator
from opstats.covariance import Covariance, CovarianceCalculator
from opstats.extended import ParallelStats, ExtendedStats, ExtendedCalculator

# Accuracy is set to 0.01, 1000 items. Result should be accurate to 1% each way.
HYPERLOGLOG_DELTA = 20
# For floats should be 0.1. For integers should be 1.
TDIGEST_DELTA = 0.1


class BaseTestCases:
    # Wrapped in a class so it isn't found by test discovery.
    class TestMoments(unittest.TestCase):
        def compare_moments(self, left: Moments, right: Moments) -> None:
            self.assertEqual(left.sample_count, right.sample_count)
            self.assertAlmostEqual(left.mean, right.mean)
            self.assertAlmostEqual(left.variance, right.variance)
            self.assertAlmostEqual(left.standard_deviation, right.standard_deviation)
            self.assertAlmostEqual(left.skewness, right.skewness)
            self.assertAlmostEqual(left.kurtosis, right.kurtosis)

        def calculate_scipy(self, data_points: List[Union[int, float]], sample_variance: bool = False, bias_adjust: bool = False) -> Moments:
            count = len(data_points)
            mean = numpy.mean(data_points)
            if sample_variance:
                var = numpy.var(data_points, ddof=1)
                sd = numpy.std(data_points, ddof=1)
            else:
                var = numpy.var(data_points)
                sd = numpy.std(data_points)

            skew = scipy.stats.skew(data_points, bias=not bias_adjust)
            if numpy.isnan(skew):
                skew = 0.0

            kurt = scipy.stats.kurtosis(data_points, bias=not bias_adjust)
            if numpy.isnan(kurt):
                kurt = -3.0

            return Moments(count, float(mean), float(var), float(sd), float(skew), float(kurt))

        def calculate(self, data_points: Union[List[int], List[float]], sample_variance: bool = False, bias_adjust: bool = False) -> Moments:
            calculator = MomentCalculator(sample_variance=sample_variance, bias_adjust=bias_adjust)
            for data_point in data_points:
                calculator.add(data_point)
            return calculator.get()

    class TestCovariance(TestMoments):
        def compare_covariance(self, left: Covariance, right: Covariance) -> None:
            super().compare_moments(left.moments_x, right.moments_x)
            super().compare_moments(left.moments_y, right.moments_y)

            self.assertEqual(left.sample_count, right.sample_count)
            # self.assertAlmostEqual(left.comoment, right.comoment)
            self.assertAlmostEqual(left.covariance, right.covariance)
            self.assertAlmostEqual(left.correlation, right.correlation)

        def calculate_scipy(self, x_data: List[Union[int, float]], y_data: List[Union[int, float]], sample_covariance: bool = False) -> Covariance:
            x_stats = super().calculate_scipy(x_data, sample_variance=sample_covariance)
            y_stats = super().calculate_scipy(y_data, sample_variance=sample_covariance)

            if sample_covariance:
                cov = numpy.cov(x_data, y_data, ddof=1)[0][1]
            else:
                cov = numpy.cov(x_data, y_data, ddof=0)[0][1]

            if numpy.isnan(cov):
                cov = 0.0

            cor = numpy.corrcoef(x_data, y_data)[0][1]

            if numpy.isnan(cor):
                cor = 0.0

            return Covariance(x_stats, y_stats, 0.0, cov, cor)

        def calculate(self, x_data: List[Union[int, float]], y_data: List[Union[int, float]], sample_covariance: bool = False) -> Covariance:
            calculator = CovarianceCalculator(sample_covariance=sample_covariance)
            for x, y in list(zip(x_data, y_data)):
                calculator.add(x, y)
            return calculator.get()

    class TestStats(TestMoments):
        def compare_stats(self, left: ExtendedStats, right: ExtendedStats, integers=False) -> None:
            self.assertEqual(left.sample_count, right.sample_count)
            self.assertAlmostEqual(left.mean, right.mean)
            self.assertAlmostEqual(left.variance, right.variance)
            self.assertAlmostEqual(left.standard_deviation, right.standard_deviation)
            self.assertAlmostEqual(left.skewness, right.skewness)
            self.assertAlmostEqual(left.kurtosis, right.kurtosis)
            self.assertAlmostEqual(left.cardinality, right.cardinality, delta=HYPERLOGLOG_DELTA)

            self.assertCountEqual(left.percentiles.keys(), right.percentiles.keys())

            if integers:
                self.assertAlmostEqual(int(left.median), int(right.median), delta=1)
                self.assertAlmostEqual(int(left.interquartile_range), int(right.interquartile_range), delta=1)
                for p in left.percentiles.keys():
                    self.assertIn(p, right.percentiles)
                    self.assertAlmostEqual(int(left.percentiles[p]), int(right.percentiles[p]), delta=1)
            else:
                self.assertAlmostEqual(left.median, right.median, delta=TDIGEST_DELTA)
                self.assertAlmostEqual(left.interquartile_range, right.interquartile_range, delta=TDIGEST_DELTA)
                for p in left.percentiles.keys():
                    self.assertIn(p, right.percentiles)
                    self.assertAlmostEqual(left.percentiles[p], right.percentiles[p], delta=TDIGEST_DELTA)

        def calculate_scipy(self, data_points: List[Union[int, float]], sample_variance: bool = False, bias_adjust: bool = False) -> ExtendedStats:
            moments = super().calculate_scipy(data_points, sample_variance=sample_variance, bias_adjust=bias_adjust)
            card = len(set(data_points))
            med = numpy.median(data_points)
            iqr = scipy.stats.iqr(data_points)
            return ExtendedStats(
                moments.sample_count,
                moments.mean,
                moments.variance,
                moments.standard_deviation,
                moments.skewness,
                moments.kurtosis,
                card,
                float(med),
                float(iqr),
                {}
            )

        def calculate(self, data_points: List[Union[int, float]], sample_variance: bool = False, bias_adjust: bool = False) -> ParallelStats:
            calculator = ExtendedCalculator(sample_variance=sample_variance, bias_adjust=bias_adjust)
            for data_point in data_points:
                calculator.add(data_point)
            return calculator.get_parallel()
