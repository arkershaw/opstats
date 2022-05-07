import unittest
from typing import List, Union

import numpy
import scipy.stats

from opstats import Stats, OnlineCalculator


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

        def calculate_scipy(self, data_points: Union[List[int], List[float]], sample_variance: bool = False, bias_adjust: bool = False) -> Stats:
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

            return Stats(count, mean, var, sd, skew, kurt)

        def calculate(self, data_points: Union[List[int], List[float]], sample_variance: bool = False, bias_adjust: bool = False) -> Stats:
            calculator = OnlineCalculator(sample_variance=sample_variance, bias_adjust=bias_adjust)
            for data_point in data_points:
                calculator.add(data_point)
            return calculator.get()
