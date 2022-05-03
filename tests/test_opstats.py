from math import sqrt
import unittest

from opstats import Stats, OnlineCalculator, aggregate_stats


SAMPLE_DATA = [
    25, 97, 76, 94, 49, 85, 65, 77, 22, 13,
    55, 99, 90, 59, 95, 79, 38, 2, 83, 60,
    21, 14, 52, 44, 48, 84, 51, 82, 72, 30,
    93, 54, 64, 89, 88, 35, 67, 28, 46, 74,
    78, 68, 16, 53, 10, 69, 66, 34, 31, 18
]


class TestOnlineCalculator(unittest.TestCase):
    def compare_stats(self, left: Stats, right: Stats) -> None:
        self.assertEqual(len(SAMPLE_DATA), left.count)
        self.assertEqual(len(SAMPLE_DATA), right.count)
        self.assertAlmostEqual(left.mean, right.mean)
        self.assertAlmostEqual(left.variance, right.variance)
        self.assertAlmostEqual(left.standard_deviation, right.standard_deviation)
        self.assertAlmostEqual(left.skew, right.skew)
        self.assertAlmostEqual(left.kurtosis, right.kurtosis)

    def write_stats(self, label: str, stats: Stats) -> None:
        print(f'{label} count: {stats.mean}')
        print(f'{label} mean: {stats.mean}')
        print(f'{label} sample variance: {stats.variance}')
        print(f'{label} standard deviation: {stats.standard_deviation}')
        print(f'{label} skew: {stats.skew}')
        print(f'{label} kurtosis: {stats.kurtosis}')
    
    def calculate_scipy(self) -> Stats:
        import scipy.stats

        count = len(SAMPLE_DATA)
        mean = scipy.stats.tmean(SAMPLE_DATA)
        s_var = scipy.stats.tvar(SAMPLE_DATA)
        sd = scipy.stats.tstd(SAMPLE_DATA)
        skew = scipy.stats.skew(SAMPLE_DATA)
        kurt = scipy.stats.kurtosis(SAMPLE_DATA)

        return Stats(count, mean, s_var, sd, skew, kurt)

    def calculate(self) -> Stats:
        calculator = OnlineCalculator()
        for data_point in SAMPLE_DATA:
            calculator.add(data_point)
        return calculator.get()

    def test_statistics(self) -> None:
        import statistics

        mean = statistics.mean(SAMPLE_DATA)
        s_var = statistics.variance(SAMPLE_DATA)
        sd = sqrt(s_var)
        # Skew and kurtosis not available.

        result = self.calculate()

        self.assertEqual(len(SAMPLE_DATA), result.count)
        self.assertAlmostEqual(mean, result.mean)
        self.assertAlmostEqual(s_var, result.variance)
        self.assertAlmostEqual(sd, result.standard_deviation)
        self.assertAlmostEqual(mean, result.mean)

    def test_scipy(self) -> None:
        scipy_result = self.calculate_scipy()
        result = self.calculate()
        self.compare_stats(scipy_result, result)

    def test_aggregate_stats(self) -> None:
        scipy_result = self.calculate_scipy()

        left_data = SAMPLE_DATA[:len(SAMPLE_DATA)//2]
        right_data = SAMPLE_DATA[len(SAMPLE_DATA)//2:]
        left = OnlineCalculator()
        for d in left_data:
            left.add(d)
        right = OnlineCalculator()
        for d in right_data:
            right.add(d)
        result = aggregate_stats([left.get(), right.get()])

        self.compare_stats(scipy_result, result)
