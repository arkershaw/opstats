import unittest
from typing import List, Union

import numpy
import scipy.stats

from opstats import Stats, OnlineCalculator, OnlineCovariance, CovarianceStats


class BaseTestCases:
    # Wrapped in a class so it isn't found by test discovery.
    class TestStats(unittest.TestCase):
        def compare_stats(self, left: Stats, right: Stats) -> None:
            self.assertEqual(left.sample_count, right.sample_count)
            self.assertAlmostEqual(left.mean, right.mean)
            self.assertAlmostEqual(left.variance, right.variance)
            self.assertAlmostEqual(left.standard_deviation, right.standard_deviation)
            self.assertAlmostEqual(left.skewness, right.skewness)
            self.assertAlmostEqual(left.kurtosis, right.kurtosis)

        def calculate_scipy(self, data_points: List[Union[int, float]], sample_variance: bool = False, bias_adjust: bool = False) -> Stats:
            count = len(data_points)
            mean = numpy.mean(data_points)
            if sample_variance:
                var = numpy.var(data_points, ddof=1)
                sd = numpy.std(data_points, ddof=1)
            else:
                var = numpy.var(data_points)
                sd = numpy.std(data_points)
            skew = scipy.stats.skew(data_points, bias=not bias_adjust)
            kurt = scipy.stats.kurtosis(data_points, bias=not bias_adjust)

            return Stats(count, float(mean), float(var), float(sd), skew, kurt)

        def calculate(self, data_points: Union[List[int], List[float]], sample_variance: bool = False, bias_adjust: bool = False) -> Stats:
            calculator = OnlineCalculator(sample_variance=sample_variance, bias_adjust=bias_adjust)
            for data_point in data_points:
                calculator.add(data_point)
            return calculator.get()

    class TestCovarianceStats(TestStats):
        def compare_stats(self, left: CovarianceStats, right: CovarianceStats) -> None:
            super().compare_stats(left.stats_x, right.stats_x)
            super().compare_stats(left.stats_y, right.stats_y)

            self.assertEqual(left.sample_count, right.sample_count)
            # self.assertAlmostEqual(left.comoment, right.comoment)
            self.assertAlmostEqual(left.covariance, right.covariance)
            self.assertAlmostEqual(left.correlation, right.correlation)

        def calculate_scipy(self, x_data: List[Union[int, float]], y_data: List[Union[int, float]], sample_covariance: bool = False) -> CovarianceStats:
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

            return CovarianceStats(x_stats, y_stats, 0.0, cov, cor)

        def calculate(self, x_data: List[Union[int, float]], y_data: List[Union[int, float]], sample_covariance: bool = False) -> CovarianceStats:
            calculator = OnlineCovariance(sample_covariance=sample_covariance)
            for x, y in list(zip(x_data, y_data)):
                calculator.add(x, y)
            return calculator.get()
