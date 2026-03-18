"""
transform_vectors
=================
 
Transform vector fields between coordinate reference systems using pyproj.
 
A pyproj-only equivalent of cartopy's ``transform_vectors``.
 
Quick start
-----------
>>> from transform_vectors import transform_vectors
>>> ut, vt = transform_vectors(src_crs, target_crs, x, y, u, v)
 
For rotated pole grids:
>>> ut, vt = transform_vectors(src_crs, target_crs, x, y, u, v,
...                            is_rotated_pole=True)
"""
 
from .transform import transform_vectors
 
__version__ = "0.1.0"
__all__      = ["transform_vectors"]