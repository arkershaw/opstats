import numpy

from opstats import Stats, OnlineCalculator

from tests.base import BaseTestCases

# Use 101 elements to get uneven sample sizes when dividing into three lists.
RANDOM_INTS = list(numpy.random.randint(1, 100, 101))
RANDOM_FLOATS = list(numpy.random.rand(101))


class TestOnlineCalculator(BaseTestCases.TestStats):
    def test_invalid_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Argument "sample_variance" must be a bool, received'):
            OnlineCalculator(sample_variance=None)  # type: ignore

        with self.assertRaisesRegex(ValueError, 'Argument "sample_variance" must be a bool, received'):
            OnlineCalculator(sample_variance='')  # type: ignore

    def test_none(self) -> None:
        empty = Stats()
        calculator = OnlineCalculator()
        calculator.add(None)  # type: ignore
        result = calculator.get()
        self.compare_stats(empty, result)

    def test_empty(self) -> None:
        empty = Stats()
        result = self.calculate([])
        self.compare_stats(empty, result)

    def test_single_value(self) -> None:
        data_points = [1.0]
        scipy_result = self.calculate_scipy(data_points)
        result = self.calculate(data_points)
        self.compare_stats(scipy_result, result)

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

    def test_sample_variance_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS, sample_variance=True)
        result = self.calculate(RANDOM_FLOATS, sample_variance=True)
        self.compare_stats(scipy_result, result)

    def test_population_variance_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS)
        result = self.calculate(RANDOM_FLOATS)
        self.compare_stats(scipy_result, result)
