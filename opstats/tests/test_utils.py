import unittest

import numpy

import opstats.utils as utils

test_params = [
    ('Even integers', list(numpy.random.randint(1, 100, 100))),
    ('Even floats', list(numpy.random.rand(100))),
    ('Odd integers', list(numpy.random.randint(1, 100, 99))),
    ('Odd floats', list(numpy.random.rand(99))),
]

METHOD = 'midpoint'


class TestUtils(unittest.TestCase):
    def test_median(self) -> None:
        for name, values in test_params:
            with self.subTest(msg=name, values=values):
                expected = numpy.percentile(values, 50, method=METHOD)
                actual = utils.percentile(values, 50, method=METHOD)
                self.assertEqual(expected, actual)

    def test_33rd_percentile(self) -> None:
        for name, values in test_params:
            with self.subTest(msg=name, values=values):
                expected = numpy.percentile(values, 100 / 3, method=METHOD)
                actual = utils.percentile(values, 100 / 3, method=METHOD)
                self.assertEqual(expected, actual)

    def test_zero(self) -> None:
        for name, values in test_params:
            with self.subTest(msg=name, values=values):
                expected = numpy.percentile(values, 0, method=METHOD)
                actual = utils.percentile(values, 0, method=METHOD)
                self.assertEqual(expected, actual)

    def test_first(self) -> None:
        for name, values in test_params:
            with self.subTest(msg=name, values=values):
                expected = numpy.percentile(values, 1, method=METHOD)
                actual = utils.percentile(values, 1, method=METHOD)
                self.assertEqual(expected, actual)

    def test_last(self) -> None:
        for name, values in test_params:
            with self.subTest(msg=name, values=values):
                expected = numpy.percentile(values, 100, method=METHOD)
                actual = utils.percentile(values, 100, method=METHOD)
                self.assertEqual(expected, actual)

    def test_to_numeric_None(self) -> None:
        expected = None
        actual = utils.to_numeric(expected)
        self.assertIsNone(actual)

    def test_to_numeric_integer(self) -> None:
        expected = 12345
        actual = utils.to_numeric(expected)
        self.assertIsInstance(actual, int)
        self.assertEqual(expected, actual)

    def test_to_numeric_float(self) -> None:
        expected = 123.45
        actual = utils.to_numeric(expected)
        self.assertIsInstance(actual, float)
        self.assertEqual(expected, actual)

    def test_to_numeric_integer_str(self) -> None:
        expected = 12345
        actual = utils.to_numeric(str(expected))
        self.assertIsInstance(actual, int)
        self.assertEqual(expected, actual)

    def test_to_numeric_float_str(self) -> None:
        expected = 123.45
        actual = utils.to_numeric(str(expected))
        self.assertIsInstance(actual, float)
        self.assertEqual(expected, actual)

    def test_to_numeric_str(self) -> None:
        expected = 'abc'
        actual = utils.to_numeric(expected)
        self.assertIsInstance(actual, int)
        self.assertEqual(len(expected), actual)

    def test_to_numeric_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Invalid type:'):
            utils.to_numeric([])
