import numpy

from opstats import CovarianceStats, OnlineCovariance, Stats

from tests.base import BaseTestCases

MEAN = [50, 50]
COV_MATRIX = [[100.0, 0], [0, 100]]
# Use 101 elements to get uneven sample sizes when dividing into three lists.
RANDOM_FLOATS = numpy.random.multivariate_normal(MEAN, COV_MATRIX, 101).T


class TestCovarianceCalculator(BaseTestCases.TestCovarianceStats):
    def test_invalid_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Argument "sample_covariance" must be a bool, received'):
            OnlineCovariance(sample_covariance=None)  # type: ignore

        with self.assertRaisesRegex(ValueError, 'Argument "sample_covariance" must be a bool, received'):
            OnlineCovariance(sample_covariance='')  # type: ignore

    def test_length_mismatch(self) -> None:
        data_points = [1.0, 1.0, 1.0]
        result = self.calculate(data_points, data_points[0:2])
        self.compare_stats(CovarianceStats(Stats(2, 1.0), Stats(2, 1.0), 0.0, 0.0, 0.0), result)

    def test_none(self) -> None:
        empty = CovarianceStats(Stats(), Stats(), 0.0, 0.0, 0.0)
        calculator = OnlineCovariance()
        calculator.add(None, None)  # type: ignore
        result = calculator.get()
        self.compare_stats(empty, result)

    def test_empty(self) -> None:
        empty = CovarianceStats(Stats(), Stats(), 0.0, 0.0, 0.0)
        result = self.calculate([], [])
        self.compare_stats(empty, result)

    def test_single_value(self) -> None:
        data_points = [1.0]
        scipy_result = self.calculate_scipy(data_points, data_points)
        result = self.calculate(data_points, data_points)
        self.compare_stats(scipy_result, result)

    def test_zeros(self) -> None:
        data_points = [0.0, 0.0]
        scipy_result = self.calculate_scipy(data_points, data_points)
        result = self.calculate(data_points, data_points)
        self.compare_stats(scipy_result, result)

    def test_ones(self) -> None:
        data_points = [1.0, 1.0, 1.0]
        scipy_result = self.calculate_scipy(data_points, data_points)
        result = self.calculate(data_points, data_points)
        self.compare_stats(scipy_result, result)

    def test_sample_variance_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS[0], RANDOM_FLOATS[1], sample_covariance=True)
        result = self.calculate(RANDOM_FLOATS[0], RANDOM_FLOATS[1], sample_covariance=True)
        self.compare_stats(scipy_result, result)

    def test_population_variance_floats(self) -> None:
        scipy_result = self.calculate_scipy(RANDOM_FLOATS[0], RANDOM_FLOATS[1])
        result = self.calculate(RANDOM_FLOATS[0], RANDOM_FLOATS[1])
        self.compare_stats(scipy_result, result)
