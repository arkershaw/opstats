from typing import List, Union

import numpy

from opstats import Stats, aggregate_stats
from tests.base import BaseTestCases

# Use 101 elements to get uneven sample sizes when dividing into three lists.
RANDOM_INTS = list(numpy.random.randint(1, 100, 101))
RANDOM_FLOATS = list(numpy.random.rand(101))


class TestAggregateStats(BaseTestCases.TestStats):
    def calculate_parallel(self, data_points: Union[List[int], List[float]], sample_variance: bool = False, bias_adjust: bool = False) -> Stats:
        # Split the list into three to get uneven sample sizes.
        size = len(data_points) // 3

        first_data = data_points[:size]
        first_stats = self.calculate(first_data, sample_variance=sample_variance, bias_adjust=bias_adjust)

        second_data = data_points[size:size * 2]
        second_stats = self.calculate(second_data, sample_variance=sample_variance, bias_adjust=bias_adjust)

        third_data = data_points[size * 2:]
        third_stats = self.calculate(third_data, sample_variance=sample_variance, bias_adjust=bias_adjust)

        return aggregate_stats([first_stats, second_stats, third_stats], sample_variance=sample_variance, bias_adjust=bias_adjust)

    def test_none(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Argument "stats" must be a list of Stats, received'):
            aggregate_stats(None)  # type: ignore

    def test_invalid_type(self) -> None:
        stats = 'Stats()'
        with self.assertRaisesRegex(ValueError, 'Argument "stats" must be a list of Stats, received'):
            aggregate_stats(stats)  # type: ignore

    def test_invalid_parameters(self) -> None:
        stats = Stats(1, 5.0)

        with self.assertRaisesRegex(ValueError, 'Argument "sample_variance" must be a bool, received'):
            aggregate_stats([stats], sample_variance=None)  # type: ignore

        with self.assertRaisesRegex(ValueError, 'Argument "sample_variance" must be a bool, received'):
            aggregate_stats([stats], sample_variance='')  # type: ignore

    def test_invalid_items(self) -> None:
        stats = [Stats(1, 5.0), 'Stats()']
        result = aggregate_stats(stats)
        self.compare_stats(stats[0], result)

        # Second item has sample count of zero.
        stats = [Stats(1, 5.0), Stats(0, 10.0)]
        result = aggregate_stats(stats)
        self.compare_stats(stats[0], result)

    def test_empty(self) -> None:
        empty = Stats()
        result = aggregate_stats([])
        self.compare_stats(empty, result)

    def test_single_value(self) -> None:
        expected = Stats(1, 5.0)
        result = aggregate_stats([expected])
        self.compare_stats(expected, result)

    def test_uniform_values(self) -> None:
        expected = Stats(2, 5.0)
        stats = [Stats(1, 5.0), Stats(1, 5.0)]
        result = aggregate_stats(stats)
        self.compare_stats(expected, result)

    def test_aggregate_sample_integers(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS, sample_variance=True)
        result = self.calculate_parallel(RANDOM_INTS, sample_variance=True)
        self.compare_stats(scipy_result, result)

    def test_aggregate_population_integers(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS)
        result = self.calculate_parallel(RANDOM_INTS)
        self.compare_stats(scipy_result, result)

    def test_aggregate_sample_bias_integers(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS, sample_variance=True, bias_adjust=True)
        result = self.calculate_parallel(RANDOM_INTS, sample_variance=True, bias_adjust=True)
        self.compare_stats(scipy_result, result)

    def test_aggregate_population_bias_integers(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS, bias_adjust=True)
        result = self.calculate_parallel(RANDOM_INTS, bias_adjust=True)
        self.compare_stats(scipy_result, result)

    def test_aggregate_sample_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS, sample_variance=True)
        result = self.calculate_parallel(RANDOM_FLOATS, sample_variance=True)
        self.compare_stats(scipy_result, result)

    def test_aggregate_population_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS)
        result = self.calculate_parallel(RANDOM_FLOATS)
        self.compare_stats(scipy_result, result)

    def test_aggregate_sample_bias_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS, sample_variance=True, bias_adjust=True)
        result = self.calculate_parallel(RANDOM_FLOATS, sample_variance=True, bias_adjust=True)
        self.compare_stats(scipy_result, result)

    def test_aggregate_population_bias_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS, bias_adjust=True)
        result = self.calculate_parallel(RANDOM_FLOATS, bias_adjust=True)
        self.compare_stats(scipy_result, result)
