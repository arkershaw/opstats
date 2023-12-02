from typing import List, Union

import numpy

from opstats import Covariance, aggregate_covariance, Moments
from opstats.tests.base import BaseTestCases

MEAN = [50, 50]
COV_MATRIX = [[100.0, 0], [0, 100]]
# Use 101 elements to get uneven sample sizes when dividing into three lists.
RANDOM_FLOATS = numpy.random.multivariate_normal(MEAN, COV_MATRIX, 101).T

# TODO: Test different length lists.


class TestAggregateCovariance(BaseTestCases.TestCovarianceStats):
    def calculate_parallel(self, data_x: List[Union[int, float]], data_y: List[Union[int, float]], sample_covariance: bool = False) -> Covariance:
        # Split the list into three to get uneven sample sizes.
        size = len(data_x) // 3

        first_x = data_x[:size]
        first_y = data_y[:size]
        first_stats = self.calculate(first_x, first_y, sample_covariance=sample_covariance)

        second_x = data_x[size:size * 2]
        second_y = data_y[size:size * 2]
        second_stats = self.calculate(second_x, second_y, sample_covariance=sample_covariance)

        third_x = data_x[size * 2:]
        third_y = data_y[size * 2:]
        third_stats = self.calculate(third_x, third_y, sample_covariance=sample_covariance)

        return aggregate_covariance([first_stats, second_stats, third_stats], sample_covariance=sample_covariance)

    def test_none(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Argument "stats" must be a list of CovarianceStats, received'):
            aggregate_covariance(None)  # type: ignore

    def test_invalid_type(self) -> None:
        stats = 'Moments()'
        with self.assertRaisesRegex(ValueError, 'Argument "stats" must be a list of CovarianceStats, received'):
            aggregate_covariance(stats)  # type: ignore

    def test_invalid_parameters(self) -> None:
        stats = Covariance(Moments(), Moments(), 0.0, 1.0, 5.0)

        with self.assertRaisesRegex(ValueError, 'Argument "sample_covariance" must be a bool, received'):
            aggregate_covariance([stats], sample_covariance=None)  # type: ignore

        with self.assertRaisesRegex(ValueError, 'Argument "sample_covariance" must be a bool, received'):
            aggregate_covariance([stats], sample_covariance='')  # type: ignore

    def test_invalid_items(self) -> None:
        stats = [Covariance(Moments(1, 1.0), Moments(1, 1.0), 0.0, 1.0, 5.0), 'Covariance()']
        result = aggregate_covariance(stats)
        self.compare_covariance(stats[0], result)

        # Second item has sample count of zero.
        stats = [Covariance(Moments(1, 1.0), Moments(1, 1.0), 0.0, 1.0, 5.0), Covariance(Moments(), Moments(), 0.0, 1.0, 5.0)]
        result = aggregate_covariance(stats)
        self.compare_covariance(stats[0], result)

    def test_empty(self) -> None:
        empty = Covariance(Moments(), Moments(), 0.0, 0.0, 0.0)
        result = aggregate_covariance([])
        self.compare_covariance(empty, result)

    def test_single_value(self) -> None:
        expected = Covariance(Moments(1, 1.0), Moments(1, 1.0), 0.0, 1.0, 5.0)
        result = aggregate_covariance([Covariance(Moments(1, 1.0), Moments(1, 1.0), 0.0, 1.0, 5.0)])
        self.compare_covariance(expected, result)

    def test_uniform_values(self) -> None:
        expected = Covariance(Moments(2, 1.0), Moments(2, 1.0), 0.0, 0.0, 0.0)
        stats = [Covariance(Moments(1, 1.0), Moments(1, 1.0), 0.0, 1.0, 5.0), Covariance(Moments(1, 1.0), Moments(1, 1.0), 0.0, 1.0, 5.0)]
        result = aggregate_covariance(stats)
        self.compare_covariance(expected, result)

    def test_aggregate_sample_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS[0], RANDOM_FLOATS[1], sample_covariance=True)
        result = self.calculate_parallel(RANDOM_FLOATS[0], RANDOM_FLOATS[1], sample_covariance=True)
        self.compare_covariance(scipy_result, result)

    def test_aggregate_population_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS[0], RANDOM_FLOATS[1])
        result = self.calculate_parallel(RANDOM_FLOATS[0], RANDOM_FLOATS[1])
        self.compare_covariance(scipy_result, result)