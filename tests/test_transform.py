"""
tests/test_transform.py
=======================

Unit tests for transform_vectors.

Run with:
    pytest tests/
"""

import numpy as np
import pytest
import pyproj
from transform_vectors import transform_vectors


# ── fixtures ──────────────────────────────────────────────────────────────────

STEREO_CRS = pyproj.CRS.from_proj4(
    "+proj=stere +lat_0=90 +lon_0=-30 +lat_ts=90 +datum=WGS84")
GEO_CRS = pyproj.CRS.from_epsg(4326)

# 2x2 grid in stereographic meters
X2 = np.array([[100000., 200000.], [100000., 200000.]])
Y2 = np.array([[-800000., -800000.], [-700000., -700000.]])

# unit vectors
E_X = np.stack([np.ones(X2.shape), np.zeros(X2.shape)], axis=-1)  # (1,0)
E_Y = np.stack([np.zeros(X2.shape), np.ones(X2.shape)],  axis=-1)  # (0,1)


# ── input validation ──────────────────────────────────────────────────────────

def test_shape_mismatch_raises():
    u = np.ones((2, 2))
    v = np.ones((2, 3))   # wrong shape
    with pytest.raises(ValueError, match="same shape"):
        transform_vectors(STEREO_CRS, GEO_CRS, X2, Y2, u, v)


def test_wrong_ndim_raises():
    u = np.ones((2, 2, 2))  # 3D not allowed
    v = np.ones((2, 2, 2))
    x = np.ones((2, 2, 2))
    y = np.ones((2, 2, 2))
    with pytest.raises(ValueError, match="1D or 2D"):
        transform_vectors(STEREO_CRS, GEO_CRS, x, y, u, v)


def test_accepts_epsg_int():
    """CRS can be passed as EPSG integer."""
    u, v = E_X[..., 0], E_X[..., 1]
    ut, vt = transform_vectors(STEREO_CRS, 4326, X2, Y2, u, v)
    assert ut.shape == u.shape


def test_accepts_proj4_string():
    """CRS can be passed as proj4 string."""
    src = "+proj=stere +lat_0=90 +lon_0=-30 +lat_ts=90 +datum=WGS84"
    u, v = E_X[..., 0], E_X[..., 1]
    ut, vt = transform_vectors(src, GEO_CRS, X2, Y2, u, v)
    assert ut.shape == u.shape


def test_accepts_1d_arrays():
    """1D arrays should work."""
    x = np.array([100000., 200000.])
    y = np.array([-800000., -700000.])
    u = np.ones(2)
    v = np.zeros(2)
    ut, vt = transform_vectors(STEREO_CRS, GEO_CRS, x, y, u, v)
    assert ut.shape == (2,)


# ── magnitude preservation ────────────────────────────────────────────────────

def test_unit_vector_magnitude_preserved():
    """Unit vectors should remain unit vectors after transform."""
    u, v = E_X[..., 0], E_X[..., 1]
    ut, vt = transform_vectors(STEREO_CRS, GEO_CRS, X2, Y2, u, v)
    norms = np.hypot(ut, vt)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_arbitrary_magnitude_preserved():
    """Magnitude of 5 should be preserved."""
    u = 5.0 * E_X[..., 0]
    v = 5.0 * E_X[..., 1]
    ut, vt = transform_vectors(STEREO_CRS, GEO_CRS, X2, Y2, u, v)
    norms = np.hypot(ut, vt)
    np.testing.assert_allclose(norms, 5.0, atol=1e-9)


def test_zero_vector_magnitude_preserved():
    """Zero vectors should remain zero."""
    u = np.zeros_like(X2)
    v = np.zeros_like(X2)
    ut, vt = transform_vectors(STEREO_CRS, GEO_CRS, X2, Y2, u, v)
    np.testing.assert_allclose(np.hypot(ut, vt), 0.0, atol=1e-10)


# ── identity transform ────────────────────────────────────────────────────────

def test_same_crs_returns_same_vectors():
    """Transforming to the same CRS should return the original vectors."""
    u, v = E_X[..., 0].copy(), E_X[..., 1].copy()
    ut, vt = transform_vectors(STEREO_CRS, STEREO_CRS, X2, Y2, u, v)
    np.testing.assert_allclose(ut, u, atol=1e-6)
    np.testing.assert_allclose(vt, v, atol=1e-6)


# ── orthogonality preservation (conformal projections) ───────────────────────

# def test_orthogonality_preserved_conformal():
#     """
#     For conformal projections (stereographic), e_x and e_y are orthogonal
#     in both source and target CRS.
#     """
#     ui, vi = E_X[..., 0], E_X[..., 1]
#     uj, vj = E_Y[..., 0], E_Y[..., 1]

#     uti, vti = transform_vectors(STEREO_CRS, GEO_CRS, X2, Y2, ui, vi)
#     utj, vtj = transform_vectors(STEREO_CRS, GEO_CRS, X2, Y2, uj, vj)

#     # dot product of transformed e_x and e_y should be ~0
#     dot = uti * utj + vti * vtj
#     np.testing.assert_allclose(dot, 0.0, atol=1e-6)


# ── south polar ───────────────────────────────────────────────────────────────

def test_south_polar():
    """South polar stereographic should work without errors."""
    src = pyproj.CRS.from_proj4(
        "+proj=stere +lat_0=-90 +lon_0=0 +lat_ts=-90 +datum=WGS84")
    x = np.array([[50000., 100000.], [50000., 100000.]])
    y = np.array([[600000., 600000.], [700000., 700000.]])
    u, v = np.ones_like(x), np.zeros_like(x)
    ut, vt = transform_vectors(src, GEO_CRS, x, y, u, v)
    np.testing.assert_allclose(np.hypot(ut, vt), 1.0, atol=1e-10)


# ── LCC ───────────────────────────────────────────────────────────────────────

def test_lcc_europe():
    """LCC Europe — mid-latitude conformal projection."""
    src = pyproj.CRS.from_proj4(
        "+proj=lcc +lat_1=43 +lat_2=62 +lat_0=52 +lon_0=10 +datum=WGS84")
    x = np.array([[-200000., 200000.], [-200000., 200000.]])
    y = np.array([[-100000., -100000.], [100000., 100000.]])
    u, v = np.ones_like(x), np.zeros_like(x)
    ut, vt = transform_vectors(src, GEO_CRS, x, y, u, v)
    np.testing.assert_allclose(np.hypot(ut, vt), 1.0, atol=1e-10)


# ── rotated pole ──────────────────────────────────────────────────────────────

def test_rotated_pole_magnitude_preserved():
    """Rotated pole: unit vectors should remain unit vectors."""
    src = pyproj.CRS.from_proj4(
        "+proj=ob_tran +o_proj=longlat +o_lat_p=45 +lon_0=270 +datum=WGS84")
    x = np.array([[-20., 20.], [-20., 20.]])
    y = np.array([[-15., -15.], [15., 15.]])
    u, v = np.ones_like(x), np.zeros_like(x)
    ut, vt = transform_vectors(src, GEO_CRS, x, y, u, v, is_rotated_pole=True)
    np.testing.assert_allclose(np.hypot(ut, vt), 1.0, atol=1e-10)


def test_rotated_pole_auto_detection():
    """is_rotated_pole should be inferred automatically for angular CRS."""
    src = pyproj.CRS.from_proj4(
        "+proj=ob_tran +o_proj=longlat +o_lat_p=45 +lon_0=270 +datum=WGS84")
    x = np.array([[-20., 20.], [-20., 20.]])
    y = np.array([[-15., -15.], [15., 15.]])
    u, v = np.ones_like(x), np.zeros_like(x)
    # without explicit flag — should still work via auto-detection
    ut, vt = transform_vectors(src, GEO_CRS, x, y, u, v)
    np.testing.assert_allclose(np.hypot(ut, vt), 1.0, atol=1e-10)


# ── dateline ──────────────────────────────────────────────────────────────────

def test_dateline_crossing():
    """Stereographic centered at dateline should not produce NaN."""
    src = pyproj.CRS.from_proj4(
        "+proj=stere +lat_0=90 +lon_0=180 +lat_ts=90 +datum=WGS84")
    x = np.array([[-200000., 200000.], [-200000., 200000.]])
    y = np.array([[-200000., -200000.], [200000., 200000.]])
    u, v = np.ones_like(x), np.zeros_like(x)
    ut, vt = transform_vectors(src, GEO_CRS, x, y, u, v)
    assert not np.any(np.isnan(ut))
    assert not np.any(np.isnan(vt))


# ── custom delta ──────────────────────────────────────────────────────────────

def test_custom_delta():
    """Custom delta should not affect magnitude."""
    u, v = E_X[..., 0], E_X[..., 1]
    for d in [0.1, 1.0, 100.0, 1000.0]:
        ut, vt = transform_vectors(STEREO_CRS, GEO_CRS, X2, Y2, u, v, delta=d)
        np.testing.assert_allclose(np.hypot(ut, vt), 1.0, atol=1e-6,
                                   err_msg=f"Failed for delta={d}")


# ── output shape ─────────────────────────────────────────────────────────────

def test_output_shape_2d():
    u, v = E_X[..., 0], E_X[..., 1]
    ut, vt = transform_vectors(STEREO_CRS, GEO_CRS, X2, Y2, u, v)
    assert ut.shape == (2, 2)
    assert vt.shape == (2, 2)


def test_output_shape_1d():
    x = np.array([100000., 200000., 300000.])
    y = np.array([-800000., -750000., -700000.])
    u = np.ones(3)
    v = np.zeros(3)
    ut, vt = transform_vectors(STEREO_CRS, GEO_CRS, x, y, u, v)
    assert ut.shape == (3,)
    assert vt.shape == (3,)