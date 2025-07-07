from functools import reduce
import math

def greatest_common_factor(numbers:list):
    if type(numbers) != list:
        raise TypeError("This data type is not accepted")
    return reduce(math.gcd, numbers)


