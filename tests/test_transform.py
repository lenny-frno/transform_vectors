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
    "+proj=stere +lat_0=90 +lon_0=-30 +lat_ts=90 +datum=WGS84"
)
GEO_CRS = pyproj.CRS.from_epsg(4326)

# 2x2 grid in stereographic meters
X2 = np.array([[100000.0, 200000.0], [100000.0, 200000.0]])
Y2 = np.array([[-800000.0, -800000.0], [-700000.0, -700000.0]])

# unit vectors
E_X = np.stack([np.ones(X2.shape), np.zeros(X2.shape)], axis=-1)  # (1,0)
E_Y = np.stack([np.zeros(X2.shape), np.ones(X2.shape)], axis=-1)  # (0,1)


# ── input validation ──────────────────────────────────────────────────────────


def test_shape_mismatch_raises():
    u = np.ones((2, 2))
    v = np.ones((2, 3))  # wrong shape
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
    x = np.array([100000.0, 200000.0])
    y = np.array([-800000.0, -700000.0])
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
        "+proj=stere +lat_0=-90 +lon_0=0 +lat_ts=-90 +datum=WGS84"
    )
    x = np.array([[50000.0, 100000.0], [50000.0, 100000.0]])
    y = np.array([[600000.0, 600000.0], [700000.0, 700000.0]])
    u, v = np.ones_like(x), np.zeros_like(x)
    ut, vt = transform_vectors(src, GEO_CRS, x, y, u, v)
    np.testing.assert_allclose(np.hypot(ut, vt), 1.0, atol=1e-10)


# ── LCC ───────────────────────────────────────────────────────────────────────


def test_lcc_europe():
    """LCC Europe — mid-latitude conformal projection."""
    src = pyproj.CRS.from_proj4(
        "+proj=lcc +lat_1=43 +lat_2=62 +lat_0=52 +lon_0=10 +datum=WGS84"
    )
    x = np.array([[-200000.0, 200000.0], [-200000.0, 200000.0]])
    y = np.array([[-100000.0, -100000.0], [100000.0, 100000.0]])
    u, v = np.ones_like(x), np.zeros_like(x)
    ut, vt = transform_vectors(src, GEO_CRS, x, y, u, v)
    np.testing.assert_allclose(np.hypot(ut, vt), 1.0, atol=1e-10)


# ── rotated pole ──────────────────────────────────────────────────────────────


def test_rotated_pole_magnitude_preserved():
    """Rotated pole: unit vectors should remain unit vectors."""
    src = pyproj.CRS.from_proj4(
        "+proj=ob_tran +o_proj=longlat +o_lat_p=45 +lon_0=270 +datum=WGS84"
    )
    x = np.array([[-20.0, 20.0], [-20.0, 20.0]])
    y = np.array([[-15.0, -15.0], [15.0, 15.0]])
    u, v = np.ones_like(x), np.zeros_like(x)
    ut, vt = transform_vectors(src, GEO_CRS, x, y, u, v, is_rotated_pole=True)
    np.testing.assert_allclose(np.hypot(ut, vt), 1.0, atol=1e-10)


def test_rotated_pole_auto_detection():
    """is_rotated_pole should be inferred automatically for angular CRS."""
    src = pyproj.CRS.from_proj4(
        "+proj=ob_tran +o_proj=longlat +o_lat_p=45 +lon_0=270 +datum=WGS84"
    )
    x = np.array([[-20.0, 20.0], [-20.0, 20.0]])
    y = np.array([[-15.0, -15.0], [15.0, 15.0]])
    u, v = np.ones_like(x), np.zeros_like(x)
    # without explicit flag — should still work via auto-detection
    ut, vt = transform_vectors(src, GEO_CRS, x, y, u, v)
    np.testing.assert_allclose(np.hypot(ut, vt), 1.0, atol=1e-10)


# ── dateline ──────────────────────────────────────────────────────────────────


def test_dateline_crossing():
    """Stereographic centered at dateline should not produce NaN."""
    src = pyproj.CRS.from_proj4(
        "+proj=stere +lat_0=90 +lon_0=180 +lat_ts=90 +datum=WGS84"
    )
    x = np.array([[-200000.0, 200000.0], [-200000.0, 200000.0]])
    y = np.array([[-200000.0, -200000.0], [200000.0, 200000.0]])
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
        np.testing.assert_allclose(
            np.hypot(ut, vt), 1.0, atol=1e-6, err_msg=f"Failed for delta={d}"
        )


# ── output shape ─────────────────────────────────────────────────────────────


def test_output_shape_2d():
    u, v = E_X[..., 0], E_X[..., 1]
    ut, vt = transform_vectors(STEREO_CRS, GEO_CRS, X2, Y2, u, v)
    assert ut.shape == (2, 2)
    assert vt.shape == (2, 2)


def test_output_shape_1d():
    x = np.array([100000.0, 200000.0, 300000.0])
    y = np.array([-800000.0, -750000.0, -700000.0])
    u = np.ones(3)
    v = np.zeros(3)
    ut, vt = transform_vectors(STEREO_CRS, GEO_CRS, x, y, u, v)
    assert ut.shape == (3,)
    assert vt.shape == (3,)


# ── save_transform ────────────────────────────────────────────────────────────

from transform_vectors.transform import save_transform, transform


class TestSaveTransform:
    """Tests for save_transform."""

    def test_output_shape(self):
        """T should have shape (len(y), len(x), 2, 2)."""
        x = np.linspace(100000.0, 300000.0, 5)
        y = np.linspace(-800000.0, -600000.0, 4)
        T = save_transform(x, y, STEREO_CRS, GEO_CRS)
        assert T.shape == (4, 5, 2, 2)

    def test_columns_are_transformed_basis_vectors(self):
        """
        Column 0 of T should equal transform_vectors applied to e_x,
        column 1 should equal transform_vectors applied to e_y.
        """
        x = np.linspace(100000.0, 200000.0, 2)
        y = np.linspace(-800000.0, -700000.0, 2)
        X, Y = np.meshgrid(x, y)

        T = save_transform(x, y, STEREO_CRS, GEO_CRS)

        u_i, v_i = np.ones(X.shape), np.zeros(X.shape)
        u_j, v_j = np.zeros(X.shape), np.ones(X.shape)

        ut_i, vt_i = transform_vectors(STEREO_CRS, GEO_CRS, X, Y, u_i, v_i)
        ut_j, vt_j = transform_vectors(STEREO_CRS, GEO_CRS, X, Y, u_j, v_j)

        np.testing.assert_allclose(T[..., 0, 0], ut_i, atol=1e-10)
        np.testing.assert_allclose(T[..., 1, 0], vt_i, atol=1e-10)
        np.testing.assert_allclose(T[..., 0, 1], ut_j, atol=1e-10)
        np.testing.assert_allclose(T[..., 1, 1], vt_j, atol=1e-10)

    # def test_consistent_with_transform_vectors(self):
    #     """
    #     Applying T via transform() should give the same result as calling
    #     transform_vectors() directly on the same vector field.
    #     """
    #     x = np.linspace(100000.0, 200000.0, 2)
    #     y = np.linspace(-800000.0, -700000.0, 2)
    #     X, Y = np.meshgrid(x, y)

    #     T = save_transform(x, y, STEREO_CRS, GEO_CRS)

    #     # arbitrary non-unit vector field
    #     u = np.random.default_rng(0).uniform(-3, 3, X.shape)
    #     v = np.random.default_rng(1).uniform(-3, 3, X.shape)

    #     # via T
    #     vec = np.stack([u, v], axis=-1)
    #     result_T = transform(T, vec)

    #     # directly — note: transform_vectors preserves magnitude, T does not
    #     # so we compare directions only
    #     ut, vt = transform_vectors(STEREO_CRS, GEO_CRS, X, Y, u, v)
    #     print(f"u origin = {u}\n")
    #     print(f"ut = {ut}\n")
    #     print(f"from saved T u = {result_T[...,0]}")
    #     angle_T = np.arctan2(result_T[..., 1], result_T[..., 0])
    #     angle_direct = np.arctan2(vt, ut)
    #     np.testing.assert_allclose(angle_T, angle_direct, atol=1e-6)
    # def test_consistent_with_transform_vectors(self):
    #     """
    #     For unit vectors, applying T via transform() should give the same
    #     direction as calling transform_vectors() directly.

    #     Note: save_transform builds a linear map from the two basis vectors.
    #     For non-unit vectors, transform() scales linearly while transform_vectors()
    #     preserves the original magnitude — these are different operations.
    #     The equivalence holds only in direction for unit inputs.
    #     """
    #     x = np.linspace(100000.0, 200000.0, 2)
    #     y = np.linspace(-800000.0, -700000.0, 2)
    #     X, Y = np.meshgrid(x, y)

    #     T = save_transform(x, y, STEREO_CRS, GEO_CRS)

    #     # test with unit vectors at various angles
    #     for angle_deg in [0, 45, 90, 135, 180, 270]:
    #         angle = np.deg2rad(angle_deg)
    #         u = np.full(X.shape, np.cos(angle))
    #         v = np.full(X.shape, np.sin(angle))

    #         # via T
    #         vec = np.stack([u, v], axis=-1)
    #         result_T = transform(T, vec)

    #         # directly
    #         ut, vt = transform_vectors(STEREO_CRS, GEO_CRS, X, Y, u, v)

    #         # compare directions
    #         angle_T = np.arctan2(result_T[..., 1], result_T[..., 0])
    #         angle_direct = np.arctan2(vt, ut)
    #         np.testing.assert_allclose(
    #             angle_T,
    #             angle_direct,
    #             atol=1e-6,
    #             err_msg=f"Direction mismatch for input angle={angle_deg}°",
    #         )

    def test_rotated_pole(self):
        """save_transform should work for rotated pole grids."""
        src = pyproj.CRS.from_proj4(
            "+proj=ob_tran +o_proj=longlat +o_lat_p=45 +lon_0=270 +datum=WGS84"
        )
        x = np.linspace(-20.0, 20.0, 3)
        y = np.linspace(-15.0, 15.0, 3)
        T = save_transform(x, y, src, GEO_CRS)
        assert T.shape == (3, 3, 2, 2)
        assert not np.any(np.isnan(T))

    def test_returns_numpy_array(self):
        """Return type must be np.ndarray."""
        x = np.linspace(100000.0, 200000.0, 2)
        y = np.linspace(-800000.0, -700000.0, 2)
        T = save_transform(x, y, STEREO_CRS, GEO_CRS)
        assert isinstance(T, np.ndarray)


# ── transform ─────────────────────────────────────────────────────────────────


class TestTransform:
    """Tests for transform (matrix-vector multiply over a field)."""

    def setup_method(self):
        """Precompute a T matrix used across multiple tests."""
        x = np.linspace(100000.0, 200000.0, 3)
        y = np.linspace(-800000.0, -700000.0, 3)
        self.T = save_transform(x, y, STEREO_CRS, GEO_CRS)  # (3, 3, 2, 2)

    def test_output_shape(self):
        """Output shape should match input vector shape."""
        u = np.stack([np.ones((3, 3)), np.zeros((3, 3))], axis=-1)
        result = transform(self.T, u)
        assert result.shape == (3, 3, 2)

    def test_identity_matrix(self):
        """Identity T should return the input unchanged."""
        ny, nx = 3, 3
        T_id = np.zeros((ny, nx, 2, 2))
        T_id[..., 0, 0] = 1.0
        T_id[..., 1, 1] = 1.0

        u = np.stack([np.ones((ny, nx)), np.ones((ny, nx)) * 2], axis=-1)
        result = transform(T_id, u)
        np.testing.assert_allclose(result, u, atol=1e-12)

    def test_zero_vector(self):
        """Zero vector should remain zero regardless of T."""
        u = np.zeros((3, 3, 2))
        result = transform(self.T, u)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_linearity(self):
        """Transform should be linear: T(a*u + b*v) == a*T(u) + b*T(v)."""
        u = np.stack([np.ones((3, 3)), np.zeros((3, 3))], axis=-1)
        v = np.stack([np.zeros((3, 3)), np.ones((3, 3))], axis=-1)
        a, b = 2.5, -1.3

        lhs = transform(self.T, a * u + b * v)
        rhs = a * transform(self.T, u) + b * transform(self.T, v)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_1d_vector_field(self):
        """transform should work on a 1D vector field with a 1D T."""
        n = 4
        T_1d = np.zeros((n, 2, 2))
        T_1d[..., 0, 0] = 1.0
        T_1d[..., 1, 1] = 1.0  # identity

        u = np.ones((n, 2))
        result = transform(T_1d, u)
        np.testing.assert_allclose(result, u, atol=1e-12)

    def test_column0_selects_first_component(self):
        """
        If T has only a non-zero first column, transform should scale u[0]
        into the output and ignore u[1].
        """
        T = np.zeros((2, 2, 2, 2))
        T[..., 0, 0] = 3.0  # first output component = 3 * u[0]
        T[..., 1, 0] = 5.0  # second output component = 5 * u[0]

        u = np.stack([np.ones((2, 2)) * 2, np.ones((2, 2)) * 99], axis=-1)
        result = transform(T, u)

        np.testing.assert_allclose(result[..., 0], 6.0, atol=1e-12)  # 3*2
        np.testing.assert_allclose(result[..., 1], 10.0, atol=1e-12)  # 5*2
