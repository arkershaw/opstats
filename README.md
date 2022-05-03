# opstats
Python implementation of an online parallel statistics calculator. This library will calculate the total, mean, variance, standard deviation, skewness and kurtosis.

Online calculation is appropriate when you don't yet have the entire dataset in order to calculate the mean (e.g. in a streaming environment). It is more processor-intensive than the traditional methods however.

When combined with parallel computation, it can also be useful when the data is very large as it works in a single pass.

## Installation

`pip install opstats`

## Usage

### Online Calculator

```
import random
from opstats import OnlineCalculator
data_points = random.sample(range(1, 100), 20)
stats = OnlineCalculator()
for d in data_points:
    stats.add(d)
result = stats.get()
```

The result will be a NamedTuple containing the computed statistics up until this point. More data can subsequently be added and the result can be retrieved again.

### Parallel Processing

Data can be split into multiple parts and processed in parallel. The resulting statistics can be combined using the `aggregate_stats` function.

```
from opstats import aggregate_stats
# Divide the sample data in half.
left_data = data_points[:len(data_points)//2]
right_data = data_points[len(data_points)//2:]
# Create stats for each half. 
left = OnlineCalculator()
for d in left_data:
    left.add(d)
right = OnlineCalculator()
for d in right_data:
    right.add(d)
# Combine the results.
result = aggregate_stats([left.get(), right.get()])
```

## Credits

Online calculator adapted from:
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
(Terriberry, Timothy B)

Aggregation translated from:
https://rdrr.io/cran/utilities/src/R/sample.decomp.R
