from numpy import arange
a = arange(16).reshape(2,2,2,2)
print a
print a[..., 0].flatten()

