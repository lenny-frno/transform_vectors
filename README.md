# transform_vectors

A **pyproj-only** equivalent of `cartopy.crs.transform_vectors`.

Transform vector fields (wind, currents, gradients...) between any two coordinate reference systems — no cartopy required.

---

## Motivation

[Cartopy](https://scitools.org.uk/cartopy/) provides `transform_vectors` to rotate vector components between projections, but it requires cartopy as a dependency. This library reimplements the same algorithm using only [pyproj](https://pyproj4.github.io/pyproj/).

This is intended as a candidate for integration into pyproj directly.
See [pyproj issue #XXXX](https://github.com/pyproj4/pyproj/issues).

---

## Algorithm

For each vector `(u, v)` at position `(x, y)` in the source CRS:

1. Compute the vector direction `θ = atan2(v, u)` and magnitude.
2. Perturb the base point by a small step `δ` in direction `θ`.
3. Project both base and perturbed points to the target CRS.
4. Recompute the angle `θ'` from the projected displacement.
5. Reconstruct the vector with the original magnitude and new angle `θ'`.

Vector **magnitudes are always preserved**. Only directions are rotated.

For **conformal projections** (stereographic, Mercator, LCC) this is exact.  
For **non-conformal projections** (LAEA, equidistant) this is an approximation — the same limitation as cartopy's implementation.

---

<!-- ## Installation

```bash
pip install transform-vectors
```

With plotting support (matplotlib + cartopy for visual validation):

```bash
pip install "transform-vectors[plot]"
``` -->

---

## Quick start

### Metric projection → lon/lat

```python
import numpy as np
import pyproj
from transform_vectors import transform_vectors

# North polar stereographic grid
src = pyproj.CRS.from_proj4(
    "+proj=stere +lat_0=90 +lon_0=-30 +lat_ts=90 +datum=WGS84")
tgt = pyproj.CRS.from_epsg(4326)   # lon/lat

# 2D grid in meters
x = np.linspace(10000, 200000, 50)
y = np.linspace(-1000000, -500000, 50)
X, Y = np.meshgrid(x, y)

# wind components in grid coordinates
u = np.ones_like(X)    # eastward in grid space
v = np.zeros_like(X)

# transform to true East/North in lon/lat space
ut, vt = transform_vectors(src, tgt, X, Y, u, v)
```

### Rotated pole grid

```python
# Rotated pole CRS — note the lon_0 = pole_longitude + 180 convention
src_rp = pyproj.CRS.from_proj4(
    "+proj=ob_tran +o_proj=longlat +o_lat_p=45 +lon_0=270 +datum=WGS84")
# equivalently: ccrs.RotatedPole(pole_longitude=90, pole_latitude=45)

x = np.linspace(-30, 30, 50)   # rotated degrees
y = np.linspace(-20, 20, 50)
X, Y = np.meshgrid(x, y)

u = np.ones_like(X)
v = np.zeros_like(X)

ut, vt = transform_vectors(src_rp, tgt, X, Y, u, v,
                            is_rotated_pole=True)
```

### CRS can be passed in any format

```python
# EPSG integer
ut, vt = transform_vectors(32632, 4326, X, Y, u, v)

# proj4 string
ut, vt = transform_vectors(
    "+proj=stere +lat_0=90 +lon_0=0 +datum=WGS84", 4326, X, Y, u, v)

# WKT string
ut, vt = transform_vectors(wkt_string, 4326, X, Y, u, v)

# pyproj.CRS object
ut, vt = transform_vectors(pyproj.CRS.from_epsg(3413), 4326, X, Y, u, v)
```

---

## API

```python
transform_vectors(
    src_crs,
    target_crs,
    x, y,
    u, v,
    delta=None,
    is_rotated_pole=False,
)
```

| Parameter | Type | Description |
|---|---|---|
| `src_crs` | any | Source CRS (EPSG int, proj4 str, WKT, or `pyproj.CRS`) |
| `target_crs` | any | Target CRS (same formats) |
| `x`, `y` | `np.ndarray` | Grid coordinates in `src_crs` units. 1D or 2D. |
| `u`, `v` | `np.ndarray` | Vector components (grid eastward, grid northward) |
| `delta` | `float` | Perturbation step. Default: `1.0` m (metric) or `0.001` ° (angular) |
| `is_rotated_pole` | `bool` | Set `True` for rotated pole grids |

**Returns** `(ut, vt)` — transformed vector components, same shape as input.

---

## Rotated pole convention

Cartopy's `RotatedPole(pole_longitude=L, pole_latitude=P)` is equivalent to the proj4 string:

```
+proj=ob_tran +o_proj=longlat +o_lat_p=P +lon_0={L+180} +datum=WGS84
```

Note the `+180` offset on `lon_0`. This is a known convention mismatch between cartopy and proj4.

---

## Visual validation

```bash
pip install "transform-vectors[plot]"

# run all 19 test cases
python examples/visual_check.py

# run specific cases
python examples/visual_check.py 1 5 15
```

Each plot shows:
- **Black** — original unit vectors in source projection  
- **Red** — transformed by `transform_vectors` (should overlap black)
- **Blue** — cartopy reference (metric CRS only, should overlap red)

---

## Running tests

```bash
pip install "transform-vectors[test]"
pytest tests/
```

---

## Comparison with cartopy

| Feature | cartopy `transform_vectors` | This library |
|---|---|---|
| Requires cartopy | ✓ | ✗ |
| Metric CRS | ✓ | ✓ |
| Rotated pole | ✓ | ✓ |
| CRS input formats | cartopy CRS only | any pyproj-compatible |
| Algorithm | perturbation | same perturbation |
| Magnitude preserved | ✓ | ✓ |
| Accuracy at the pole| bad | better|

---

## Contributing

Contributions are welcome. Please open an issue before submitting a PR.

This project targets eventual integration into [pyproj](https://github.com/pyproj4/pyproj). Please follow pyproj's code style (`ruff`) and testing conventions.

---

## License

MIT