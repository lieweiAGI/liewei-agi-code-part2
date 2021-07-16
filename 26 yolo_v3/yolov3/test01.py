import numpy as np
import math

_boxes = np.array([1, 12 ,13, 51, 18, 2 ,22, 31, 55, 98, 2, 44, 33, 62, 62])
print(np.stack(np.split(_boxes, len(_boxes) // 5)))

a = {}
a[3] = 12
print(a)
a[4] = 13
print(a)

print(math.modf(4.3))

a = np.arange(24).reshape(2,4,3)
print(a)
print(a[...,0])