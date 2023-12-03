import numpy

from opstats.extended import ExtendedStats, ExtendedCalculator
from opstats.tests.base import BaseTestCases

# Use an even number of elements to get uneven sample sizes when dividing into three lists.
RANDOM_INTS = list(numpy.random.randint(1, 100, 1000))
RANDOM_FLOATS = list(numpy.random.rand(1000))


class TestStatsCalculator(BaseTestCases.TestStats):
    def test_invalid_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Argument "sample_variance" must be a bool, received'):
            ExtendedCalculator(sample_variance=None)  # type: ignore

        with self.assertRaisesRegex(ValueError, 'Argument "sample_variance" must be a bool, received'):
            ExtendedCalculator(sample_variance='')  # type: ignore

        with self.assertRaisesRegex(ValueError, 'Argument "bias_adjust" must be a bool, received'):
            ExtendedCalculator(bias_adjust=None)  # type: ignore

        with self.assertRaisesRegex(ValueError, 'Argument "bias_adjust" must be a bool, received'):
            ExtendedCalculator(bias_adjust='')  # type: ignore

    def test_none(self) -> None:
        empty = ExtendedStats()
        calculator = ExtendedCalculator()
        calculator.add(None)  # type: ignore
        result = calculator.get()
        self.compare_stats(empty, result)

    def test_empty(self) -> None:
        empty = ExtendedStats()
        result = self.calculate([]).calculate()
        self.compare_stats(empty, result)

    def test_single_value(self) -> None:
        data_points = [1.0]
        scipy_result = self.calculate_scipy(data_points)
        result = self.calculate(data_points).calculate()
        self.compare_stats(scipy_result, result)

    def test_zeros(self) -> None:
        data_points = [0.0, 0.0]
        scipy_result = self.calculate_scipy(data_points)
        result = self.calculate(data_points).calculate()
        self.compare_stats(scipy_result, result)

    def test_ones(self) -> None:
        data_points = [1.0, 1.0, 1.0]
        scipy_result = self.calculate_scipy(data_points)
        result = self.calculate(data_points).calculate()
        self.compare_stats(scipy_result, result)

    def test_sample_variance_integer(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS, sample_variance=True)
        result = self.calculate(RANDOM_INTS, sample_variance=True).calculate()
        self.compare_stats(scipy_result, result, True)

    def test_population_variance_integers(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS)
        result = self.calculate(RANDOM_INTS).calculate()
        self.compare_stats(scipy_result, result, True)

    def test_sample_bias_integers(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS, sample_variance=True, bias_adjust=True)
        result = self.calculate(RANDOM_INTS, sample_variance=True, bias_adjust=True).calculate()
        self.compare_stats(scipy_result, result, True)

    def test_population_bias_integers(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_INTS, bias_adjust=True)
        result = self.calculate(RANDOM_INTS, bias_adjust=True).calculate()
        self.compare_stats(scipy_result, result, True)

    def test_sample_variance_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS, sample_variance=True)
        result = self.calculate(RANDOM_FLOATS, sample_variance=True).calculate()
        self.compare_stats(scipy_result, result)

    def test_population_variance_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS)
        result = self.calculate(RANDOM_FLOATS).calculate()
        self.compare_stats(scipy_result, result)

    def test_sample_bias_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS, sample_variance=True, bias_adjust=True)
        result = self.calculate(RANDOM_FLOATS, sample_variance=True, bias_adjust=True).calculate()
        self.compare_stats(scipy_result, result)

    def test_population_bias_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS, bias_adjust=True)
        result = self.calculate(RANDOM_FLOATS, bias_adjust=True).calculate()
        self.compare_stats(scipy_result, result)
