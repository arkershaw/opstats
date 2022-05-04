from typing import List, Union
import unittest

import numpy
import scipy.stats

from opstats import Stats, OnlineCalculator, aggregate_stats, InvalidStateException

# Use 101 elements to get uneven sample sizes when dividing into three lists.
RANDOM_INTS = list(numpy.random.randint(1, 100, 101))
RANDOM_FLOATS = list(numpy.random.rand(101))


class TestOnlineCalculator(unittest.TestCase):
    def compare_stats(self, left: Stats, right: Stats) -> None:
        self.assertEqual(left.sample_count, right.sample_count)
        self.assertAlmostEqual(left.mean, right.mean)
        self.assertAlmostEqual(left.variance, right.variance)
        self.assertAlmostEqual(left.standard_deviation, right.standard_deviation)
        self.assertAlmostEqual(left.skewness, right.skewness)
        self.assertAlmostEqual(left.kurtosis, right.kurtosis)

    def calculate_scipy(self, data_points: Union[List[int], List[float]], sample_variance: bool = False) -> Stats:
        count = len(data_points)
        mean = numpy.mean(data_points)
        if sample_variance:
            var = numpy.var(data_points, ddof=1)
            sd = numpy.std(data_points, ddof=1)
        else:
            var = numpy.var(data_points)
            sd = numpy.std(data_points)
        skew = scipy.stats.skew(data_points)
        kurt = scipy.stats.kurtosis(data_points)

        return Stats(count, mean, var, sd, skew, kurt)

    def calculate(self, data_points: Union[List[int], List[float]], sample_variance: bool = False) -> Stats:
        calculator = OnlineCalculator(sample_variance=sample_variance)
        for data_point in data_points:
            calculator.add(data_point)
        return calculator.get()

    def calculate_parallel(self, data_points: Union[List[int], List[float]], sample_variance: bool = False) -> Stats:
        # Split the list into three to get uneven sample sizes.
        size = len(data_points) // 3

        first_data = data_points[:size]
        first_calc = OnlineCalculator(sample_variance=sample_variance)
        for d in first_data:
            first_calc.add(d)

        second_data = data_points[size:size * 2]
        second_calc = OnlineCalculator(sample_variance=sample_variance)
        for d in second_data:
            second_calc.add(d)

        third_data = data_points[size * 2:]
        third_calc = OnlineCalculator(sample_variance=sample_variance)
        for d in third_data:
            third_calc.add(d)

        return aggregate_stats([first_calc.get(), second_calc.get(), third_calc.get()], sample_variance=sample_variance)

    def test_empty(self) -> None:
        data_points = []
        with self.assertRaisesRegex(InvalidStateException, 'At least two data points must be added.'):
            self.calculate(data_points)

    def test_single_value(self) -> None:
        data_points = [1.0]
        with self.assertRaisesRegex(InvalidStateException, 'At least two data points must be added.'):
            self.calculate(data_points)

    def test_zeros(self) -> None:
        data_points = [0.0, 0.0]
        scipy_result = self.calculate_scipy(data_points)
        result = self.calculate(data_points)
        self.compare_stats(scipy_result, result)

    def test_ones(self) -> None:
        data_points = [1.0, 1.0, 1.0]
        scipy_result = self.calculate_scipy(data_points)
        result = self.calculate(data_points)
        self.compare_stats(scipy_result, result)

    def test_sample_variance_integer(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS, sample_variance=True)
        result = self.calculate(RANDOM_INTS, sample_variance=True)
        self.compare_stats(scipy_result, result)

    def test_population_variance_integers(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS)
        result = self.calculate(RANDOM_INTS)
        self.compare_stats(scipy_result, result)

    def test_aggregate_sample_integers(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS, sample_variance=True)
        print(f'Sample variance: {scipy_result.variance}')
        result = self.calculate_parallel(RANDOM_INTS, sample_variance=True)
        self.compare_stats(scipy_result, result)

    def test_aggregate_population_integers(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS)
        print(f'Population variance: {scipy_result.variance}')
        result = self.calculate_parallel(RANDOM_INTS)
        self.compare_stats(scipy_result, result)

    def test_sample_variance_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS, sample_variance=True)
        result = self.calculate(RANDOM_FLOATS, sample_variance=True)
        self.compare_stats(scipy_result, result)

    def test_population_variance_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS)
        result = self.calculate(RANDOM_FLOATS)
        self.compare_stats(scipy_result, result)

    def test_aggregate_sample_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS, sample_variance=True)
        result = self.calculate_parallel(RANDOM_FLOATS, sample_variance=True)
        self.compare_stats(scipy_result, result)

    def test_aggregate_population_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS)
        result = self.calculate_parallel(RANDOM_FLOATS)
        self.compare_stats(scipy_result, result)
