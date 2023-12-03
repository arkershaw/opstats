# opstats
Python implementation of an online parallel statistics calculator. This library will calculate the total, mean, variance, standard deviation, skewness and kurtosis. There are additional options for calculating covariance and correlation between two sequences of data points.

Online calculation is appropriate when you don't yet have the entire dataset in order to calculate the mean (e.g. in a streaming environment). It is more processor-intensive than the traditional methods however.

When combined with parallel computation, it can also be useful when the data is very large as it works in a single pass and can be distributed.

## Installation

`pip install opstats`

## Usage

### Moment Calculator

For calculating the mean, variance (and standard deviation), skewness and kurtosis, use the MomentCalculator.

```
import random
from opstats import MomentCalculator
data_points = random.sample(range(1, 100), 20)
calc = MomentCalculator()
for d in data_points:
    calc.add(d)

result = calc.get()
```

The result will be a NamedTuple containing the computed moments up until this point. More data can subsequently be added and the result can be retrieved again.

### Parallel Processing

Data can be split into multiple parts and processed in parallel. The resulting statistics can be combined using the `aggregate_moments` function.

```
from opstats import aggregate_moments
# Divide the sample data in half.
left_data = data_points[:len(data_points)//2]
right_data = data_points[len(data_points)//2:]
# Create stats for each half. 
left = MomentCalculator()
for d in left_data:
    left.add(d)

right = MomentCalculator()
for d in right_data:
    right.add(d)

# Combine the results.
result = aggregate_moments([left.get(), right.get()])
```

### Covariance and Correlation

The `CovarianceCalculator` class and `aggregate_covariance` function work in the same manner as above for calculating the covariance and correlation between two sequences of data points.

### Extended Statistics

When installed with `pip install opstats[extended]`, cardinality and percentiles can also be estimated. Cardinality will be estimated with HyperLogLog and percentiles with T-Digest.

```
from opstats.extended import ExtendedCalculator
calc = ExtendedCalculator()
for d in data_points:
    calc.add(d)

result = calc.get()
```

This can also be calculated in parallel. Note the changes to using `get_parallel()` which returns an intermediate object and `calculate()` which computes the final values.

```
from opstats.extended import aggregate_extended
# Divide the sample data in half.
left_data = data_points[:len(data_points)//2]
right_data = data_points[len(data_points)//2:]
# Create stats for each half. 
left = ExtendedCalculator()
for d in left_data:
    left.add(d)

right = ExtendedCalculator()
for d in right_data:
    right.add(d)

# Combine the results.
result = aggregate_extended([left.get_parallel(), right.get_parallel()]).calculate()
```

## Credits

Online calculator adapted from:
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
(Terriberry, Timothy B)

Aggregation translated from:
https://rdrr.io/cran/utilities/src/R/sample.decomp.R

Python HyperLogLog implementation:
https://github.com/svpcom/hyperloglog

Python T-Digest implementation:
https://github.com/CamDavidsonPilon/tdigest
