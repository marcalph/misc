""" swap values
"""
print("swap", end="\n****\n")
a = 5
b = 10
a, b = b, a
print(a) # 10
print(b) # 5
print("**********")

""" mem usage
"""
print("mem usage", end="\n*********\n")
import sys
print(sys.getsizeof(10))
print("**********")

""" time
"""
print("time execution", end="\n**************\n")
import time
start_time = time.time()
a,b = 5,10
c = a+b
end_time = time.time()
time_taken = (end_time - start_time)
print("Time taken:", time_taken)
print("**********")


""" map
"""

print("map function", end="\n************\n")
from log import logthis
import logging
logger = logging.getLogger('snippets')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s \n %(message)s", "%Y-%m-%d %H:%M")
ch.setFormatter(formatter)
logger.addHandler(ch)


@logthis
def square(n):
    return n * n


obj = (1, 2, 3)
result = map(square, obj)
print(list(result)) # {1, 4, 9}
print("**********")


""" filter
"""
print("filter function", end="\n***************\n")
arr = [1, 2, 3, 4, 5]
arr = list(filter(lambda x : x%2 == 0, arr))
print (arr) # [2, 4]
print("**********")
