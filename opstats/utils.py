from typing import List, Union, Optional
import math

def percentile(values: List[Union[int, float]], percentile: int, method: str = 'midpoint') -> float:
    if percentile < 0 or percentile > 100:
        raise ValueError(f'Argument "percentile" must be a positive integer between 0 and 100, received {percentile}')

    if len(values) == 0:
        return 0
    else:
        sv = sorted(values)

    if percentile == 0:
        return sv[0]
    elif percentile == 100:
        return sv[-1]
    else:
        ix = (percentile / 100) * (len(sv) - 1)

        if method == 'higher':
            return sv[math.ceil(ix)]
        elif method == 'midpoint':
            if ix.is_integer():
                return sv[int(ix)]
            else:
                return (sv[int(ix)] + sv[int(ix) + 1]) / 2
