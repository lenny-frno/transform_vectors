"""
tests/test_cases.py
===================

Shared projection test cases used for both automated tests and visual examples.

Each test case defines a source projection, cartopy CRS for plotting,
a grid extent, and notes on what is being tested.

The ``x`` and ``y`` arrays stored here define the **spatial extent** of the
grid only (i.e. ``x = [x_min, x_max]``).  The number of grid points is
controlled separately at runtime (see ``build_grid`` below and the
``--npts`` option in ``examples/visual_check.py``).
"""

import numpy as np


def build_grid(tc, npts=2):
    """
    Build a 2D meshgrid for a test case with a given number of points
    per axis.

    Parameters
    ----------
    tc   : dict, one entry from TEST_CASES
    npts : int, number of points along each axis (default 2 = corners only)

    Returns
    -------
    X, Y : 2D np.ndarray of shape (npts, npts)
    x, y : 1D np.ndarray of length npts
    """
    x = np.linspace(tc["x"][0], tc["x"][-1], npts)
    y = np.linspace(tc["y"][0], tc["y"][-1], npts)
    X, Y = np.meshgrid(x, y)
    return X, Y, x, y


def rotated_pole_case(name, pole_lon, pole_lat, x, y, pad=5, note=""):
    """
    Helper to build a consistent rotated pole test case.

    Cartopy's RotatedPole uses pole_longitude as lon_0 internally.
    The equivalent proj4 string requires lon_0 = pole_longitude + 180.

    Parameters
    ----------
    name     : str
    pole_lon : float, pole longitude for cartopy's RotatedPole
    pole_lat : float, pole latitude  for cartopy's RotatedPole
    x        : array-like, [x_min, x_max] in rotated degrees
    y        : array-like, [y_min, y_max] in rotated degrees
    pad      : float, padding in degrees
    note     : str, short description of what is being tested
    """
    return {
        "name": name,
        "proj4": (
            f"+proj=ob_tran +o_proj=longlat "
            f"+o_lat_p={pole_lat} +lon_0={pole_lon + 180} +datum=WGS84"
        ),
        "cartopy_kwargs": {
            "cls": "RotatedPole",
            "pole_longitude": pole_lon,
            "pole_latitude": pole_lat,
        },
        "x": np.array([x[0], x[-1]], dtype=float),
        "y": np.array([y[0], y[-1]], dtype=float),
        "pad": pad,
        "is_rotated_pole": True,
        "note": note,
    }


TEST_CASES = [
    # ── Polar stereographic ───────────────────────────────────────────────────
    {
        "name": "01 - North Polar Stereographic (lon_0=-30)",
        "proj4": "+proj=stere +lat_0=90 +lon_0=-30 +lat_ts=90 +datum=WGS84",
        "cartopy_kwargs": {
            "cls": "Stereographic",
            "central_latitude": 90,
            "central_longitude": -30,
            "true_scale_latitude": 90,
        },
        "x": np.array([10000., 200000.]),
        "y": np.array([-1000000., -500000.]),
        "pad": 200000,
        "is_rotated_pole": False,
        "note": "Baseline conformal polar case",
    },
    {
        "name": "02 - North Polar Stereographic far from center",
        "proj4": "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=90 +datum=WGS84",
        "cartopy_kwargs": {
            "cls": "Stereographic",
            "central_latitude": 90,
            "central_longitude": 0,
            "true_scale_latitude": 90,
        },
        "x": np.array([500000., 1500000.]),
        "y": np.array([-2000000., -1000000.]),
        "pad": 300000,
        "is_rotated_pole": False,
        "note": "Grid far from projection center — larger rotation angle",
    },
    {
        "name": "03 - South Polar Stereographic",
        "proj4": "+proj=stere +lat_0=-90 +lon_0=0 +lat_ts=-90 +datum=WGS84",
        "cartopy_kwargs": {
            "cls": "Stereographic",
            "central_latitude": -90,
            "central_longitude": 0,
            "true_scale_latitude": -90,
        },
        "x": np.array([10000., 200000.]),
        "y": np.array([500000., 1000000.]),
        "pad": 200000,
        "is_rotated_pole": False,
        "note": "Southern hemisphere — sign flip test",
    },
    {
        "name": "04 - Polar grid crossing the pole",
        "proj4": "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=90 +datum=WGS84",
        "cartopy_kwargs": {
            "cls": "Stereographic",
            "central_latitude": 90,
            "central_longitude": 0,
            "true_scale_latitude": 90,
        },
        "x": np.array([-100000., 100000.]),
        "y": np.array([-100000., 100000.]),
        "pad": 200000,
        "is_rotated_pole": False,
        "note": "Grid straddles origin — crosses the pole itself",
    },
    # ── Mid-latitude conformal ────────────────────────────────────────────────
    {
        "name": "05 - Lambert Conformal Conic (Europe)",
        "proj4": "+proj=lcc +lat_1=43 +lat_2=62 +lat_0=52 +lon_0=10 +datum=WGS84",
        "cartopy_kwargs": {
            "cls": "LambertConformal",
            "central_longitude": 10,
            "central_latitude": 52,
            "standard_parallels": (43, 62),
        },
        "x": np.array([-500000., 500000.]),
        "y": np.array([-300000., 300000.]),
        "pad": 200000,
        "is_rotated_pole": False,
        "note": "Mid-latitude conformal",
    },
    {
        "name": "06 - Lambert Conformal Conic (North America)",
        "proj4": "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +datum=WGS84",
        "cartopy_kwargs": {
            "cls": "LambertConformal",
            "central_longitude": -96,
            "central_latitude": 39,
            "standard_parallels": (33, 45),
        },
        "x": np.array([-500000., 500000.]),
        "y": np.array([-300000., 300000.]),
        "pad": 200000,
        "is_rotated_pole": False,
        "note": "Different standard parallels from Europe case",
    },
    {
        "name": "07 - Continental scale LCC (all North America)",
        "proj4": "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +datum=WGS84",
        "cartopy_kwargs": {
            "cls": "LambertConformal",
            "central_longitude": -96,
            "central_latitude": 39,
            "standard_parallels": (33, 45),
        },
        "x": np.array([-4000000., 4000000.]),
        "y": np.array([-2000000., 2000000.]),
        "pad": 500000,
        "is_rotated_pole": False,
        "note": "8000 km span — Jacobian varies strongly across domain",
    },
    # ── Mercator ──────────────────────────────────────────────────────────────
    {
        "name": "08 - Mercator mid-latitude",
        "proj4": "+proj=merc +lat_ts=45 +lon_0=10 +datum=WGS84",
        "cartopy_kwargs": {
            "cls": "Mercator",
            "central_longitude": 10,
            "latitude_true_scale": 45,
        },
        "x": np.array([-200000., 200000.]),
        "y": np.array([4000000., 5000000.]),
        "pad": 200000,
        "is_rotated_pole": False,
        "note": "Conformal, strong N/S metric stretch",
    },
    {
        "name": "09 - Mercator equatorial",
        "proj4": "+proj=merc +lon_0=0 +datum=WGS84",
        "cartopy_kwargs": {"cls": "Mercator", "central_longitude": 0},
        "x": np.array([-200000., 200000.]),
        "y": np.array([-200000., 200000.]),
        "pad": 200000,
        "is_rotated_pole": False,
        "note": "Near-identity transform — arrows should be nearly axis-aligned",
    },
    {
        "name": "10 - Mercator crossing the dateline",
        "proj4": "+proj=merc +lon_0=180 +datum=WGS84",
        "cartopy_kwargs": {"cls": "Mercator", "central_longitude": 180},
        "x": np.array([-200000., 200000.]),
        "y": np.array([-200000., 200000.]),
        "pad": 200000,
        "is_rotated_pole": False,
        "note": "lon wraps ±180 — finite differences may flip sign",
    },
    # ── Equal-area ────────────────────────────────────────────────────────────
    {
        "name": "11 - Lambert Azimuthal Equal-Area (Europe)",
        "proj4": "+proj=laea +lat_0=52 +lon_0=10 +datum=WGS84",
        "cartopy_kwargs": {
            "cls": "LambertAzimuthalEqualArea",
            "central_longitude": 10,
            "central_latitude": 52,
        },
        "x": np.array([-500000., 500000.]),
        "y": np.array([-300000., 300000.]),
        "pad": 200000,
        "is_rotated_pole": False,
        "note": "Non-conformal — rotation-only is an approximation here",
    },
    {
        "name": "12 - LAEA centered on equator (maximum shear)",
        "proj4": "+proj=laea +lat_0=0 +lon_0=0 +datum=WGS84",
        "cartopy_kwargs": {
            "cls": "LambertAzimuthalEqualArea",
            "central_longitude": 0,
            "central_latitude": 0,
        },
        "x": np.array([-500000., 500000.]),
        "y": np.array([-500000., 500000.]),
        "pad": 200000,
        "is_rotated_pole": False,
        "note": "Maximum shear in equal-area projection",
    },
    {
        "name": "13 - Oblique Mercator (45 degree azimuth)",
        "proj4": "+proj=omerc +lat_0=45 +lonc=10 +alpha=45 +datum=WGS84",
        "cartopy_kwargs": {
            "cls": "ObliqueMercator",
            "central_longitude": 10,
            "central_latitude": 45,
            "azimuth": 45,
        },
        "x": np.array([-500000., 500000.]),
        "y": np.array([-300000., 300000.]),
        "pad": 200000,
        "is_rotated_pole": False,
        "note": "Grid axes rotated 45 from E/N — strong off-diagonal Jacobian",
    },
    # ── Rotated pole ──────────────────────────────────────────────────────────
    rotated_pole_case(
        "14 - Rotated Pole (Europe, pole at 30N)",
        pole_lon=170, pole_lat=30,
        x=[-20., 20.], y=[-15., 15.],
        note="Standard European rotated pole",
    ),
    rotated_pole_case(
        "15 - Rotated Pole (Nordic, pole at 15N)",
        pole_lon=160, pole_lat=15,
        x=[-10., 10.], y=[-10., 10.],
        note="Nordic rotated pole",
    ),
    rotated_pole_case(
        "16 - Rotated Pole (strong rotation, pole at 45N)",
        pole_lon=90, pole_lat=45,
        x=[-30., 30.], y=[-20., 20.],
        note="Strong rotation",
    ),
    rotated_pole_case(
        "17 - Rotated Pole (pole ON equator, maximum distortion)",
        pole_lon=0, pole_lat=0,
        x=[-30., 30.], y=[-30., 30.],
        note="Pole on equator — extreme case",
    ),
    rotated_pole_case(
        "18 - Rotated Pole over the Pacific",
        pole_lon=-160, pole_lat=20,
        x=[-40., 40.], y=[-20., 20.],
        note="Pacific domain — tests lon wraparound with rotated pole",
    ),
]