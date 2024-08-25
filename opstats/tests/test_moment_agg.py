from typing import List, Union

import numpy

from opstats.moments import Moments, aggregate_moments
from opstats.tests.base import BaseTestCases

# Use an even number of elements to get uneven sample sizes when dividing into three lists.
RANDOM_INTS = [int(v) for v in numpy.random.randint(1, 100, 1000)]
RANDOM_FLOATS = [float(v) for v in numpy.random.rand(1000)]


class TestAggregateMoments(BaseTestCases.TestMoments):
    def calculate_parallel(self, data_points: Union[List[int], List[float]], sample_variance: bool = False, bias_adjust: bool = False) -> Moments:
        # Split the list into three to get uneven sample sizes.
        size = len(data_points) // 3

        first_data = data_points[:size]
        first_moments = self.calculate(first_data, sample_variance=sample_variance, bias_adjust=bias_adjust)

        second_data = data_points[size:size * 2]
        second_moments = self.calculate(second_data, sample_variance=sample_variance, bias_adjust=bias_adjust)

        third_data = data_points[size * 2:]
        third_moments = self.calculate(third_data, sample_variance=sample_variance, bias_adjust=bias_adjust)

        return aggregate_moments([first_moments, second_moments, third_moments], sample_variance=sample_variance, bias_adjust=bias_adjust)

    def test_none(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Argument "moments" must be a list of Moments, received'):
            aggregate_moments(None)  # type: ignore

    def test_invalid_type(self) -> None:
        moments = 'Moments()'
        with self.assertRaisesRegex(ValueError, 'Argument "moments" must be a list of Moments, received'):
            aggregate_moments(moments)  # type: ignore

    def test_invalid_parameters(self) -> None:
        moments = Moments(1, 5.0)

        with self.assertRaisesRegex(ValueError, 'Argument "sample_variance" must be a bool, received'):
            aggregate_moments([moments], sample_variance=None)  # type: ignore

        with self.assertRaisesRegex(ValueError, 'Argument "sample_variance" must be a bool, received'):
            aggregate_moments([moments], sample_variance='')  # type: ignore

    def test_invalid_items(self) -> None:
        moments = [Moments(1, 5.0), 'Moments()']
        result = aggregate_moments(moments)
        self.compare_moments(moments[0], result)

        # Second item has sample count of zero.
        moments = [Moments(1, 5.0), Moments(0, 10.0)]
        result = aggregate_moments(moments)
        self.compare_moments(moments[0], result)

    def test_empty(self) -> None:
        empty = Moments()
        result = aggregate_moments([])
        self.compare_moments(empty, result)

    def test_single_value(self) -> None:
        expected = Moments(1, 5.0)
        result = aggregate_moments([expected])
        self.compare_moments(expected, result)

    def test_uniform_values(self) -> None:
        expected = Moments(2, 5.0)
        moments = [Moments(1, 5.0), Moments(1, 5.0)]
        result = aggregate_moments(moments)
        self.compare_moments(expected, result)

    def test_aggregate_sample_integers(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS, sample_variance=True)
        result = self.calculate_parallel(RANDOM_INTS, sample_variance=True)
        self.compare_moments(scipy_result, result)

    def test_aggregate_population_integers(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS)
        result = self.calculate_parallel(RANDOM_INTS)
        self.compare_moments(scipy_result, result)

    def test_aggregate_population_integer_strings(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS)
        result = self.calculate_parallel([str(v) for v in RANDOM_INTS])
        self.compare_moments(scipy_result, result)

    def test_aggregate_sample_bias_integers(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS, sample_variance=True, bias_adjust=True)
        result = self.calculate_parallel(RANDOM_INTS, sample_variance=True, bias_adjust=True)
        self.compare_moments(scipy_result, result)

    def test_aggregate_population_bias_integers(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS, bias_adjust=True)
        result = self.calculate_parallel(RANDOM_INTS, bias_adjust=True)
        self.compare_moments(scipy_result, result)

    def test_aggregate_sample_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS, sample_variance=True)
        result = self.calculate_parallel(RANDOM_FLOATS, sample_variance=True)
        self.compare_moments(scipy_result, result)

    def test_aggregate_population_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS)
        result = self.calculate_parallel(RANDOM_FLOATS)
        self.compare_moments(scipy_result, result)

    def test_aggregate_population_float_strings(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS)
        result = self.calculate_parallel([float(v) for v in RANDOM_FLOATS])
        self.compare_moments(scipy_result, result)

    def test_aggregate_sample_bias_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS, sample_variance=True, bias_adjust=True)
        result = self.calculate_parallel(RANDOM_FLOATS, sample_variance=True, bias_adjust=True)
        self.compare_moments(scipy_result, result)

    def test_aggregate_population_bias_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS, bias_adjust=True)
        result = self.calculate_parallel(RANDOM_FLOATS, bias_adjust=True)
        self.compare_moments(scipy_result, result)

    def test_aggregate_population_mixed(self) -> None:
        scipy_result = self.calculate_scipy([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
        result = self.calculate_parallel([1, 1.5, '2', '2.5', 'abc', 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
        self.compare_moments(scipy_result, result)
