"""
transform_vectors.transform
===========================

Transform vector fields between coordinate reference systems using pyproj.

This module provides a pyproj-only equivalent of cartopy's
``transform_vectors`` function, with additional support for rotated pole
grids. It uses the same perturbation algorithm as cartopy but removes the
cartopy dependency entirely.

Algorithm
---------
For each vector (u, v) at position (x, y) in the source CRS:

1. Compute the vector direction angle θ = atan2(v, u) and magnitude.
2. Perturb the base point by a small step δ in the direction θ.
3. Project both base and perturbed points to the target CRS.
4. Recompute the angle θ' from the projected displacement.
5. Reconstruct the vector with the original magnitude and new angle θ'.

References
----------
Equivalent to ``cartopy.crs.CRS.transform_vectors``:
https://scitools.org.uk/cartopy/docs/latest/reference/crs.html
"""

import warnings

import numpy as np
from pyproj import CRS, Transformer


__all__ = ["transform_vectors"]


def transform_vectors(
    src_crs,
    target_crs,
    x,
    y,
    u,
    v,
    delta=None,
    is_rotated_pole=False,
):
    """
    Transform vector components from one CRS to another.

    Uses a perturbation algorithm to rotate vector directions from the
    source CRS to the target CRS while preserving vector magnitudes.
    Works with any CRS supported by pyproj, including rotated pole grids.

    Parameters
    ----------
    src_crs : any
        Source CRS. Accepts any format supported by ``pyproj.CRS.from_user_input``:
        EPSG code (int or ``"EPSG:4326"``), proj4 string, WKT, or
        ``pyproj.CRS`` object.
    target_crs : any
        Target CRS. Same formats as ``src_crs``.
    x : np.ndarray
        X coordinates of the vector base points in ``src_crs`` units.
        1D or 2D.
    y : np.ndarray
        Y coordinates of the vector base points in ``src_crs`` units.
        Same shape as ``x``.
    u : np.ndarray
        Vector components in the x-direction (grid eastward).
        Same shape as ``x``.
    v : np.ndarray
        Vector components in the y-direction (grid northward).
        Same shape as ``x``.
    delta : float, optional
        Perturbation step in source CRS native units.
        Defaults to ``1.0`` for metric CRS (meters) and ``0.001`` for
        angular CRS (degrees). Increase for smoother results on coarse
        grids, decrease for higher accuracy on fine grids.
    is_rotated_pole : bool, optional
        Set to ``True`` for rotated pole grids. This forces degree-based
        perturbation and correct domain boundary handling regardless of
        what ``src_crs.axis_info`` reports. Default is ``False``.

    Returns
    -------
    ut : np.ndarray
        Transformed vector x-components in ``target_crs`` units.
        Same shape as input ``u``.
    vt : np.ndarray
        Transformed vector y-components in ``target_crs`` units.
        Same shape as input ``v``.

    Raises
    ------
    ValueError
        If ``x``, ``y``, ``u``, ``v`` do not all have the same shape.
    ValueError
        If arrays are not 1D or 2D.

    Warns
    -----
    UserWarning
        If some vectors at domain corners cannot be correctly transformed
        (same behaviour as cartopy's ``transform_vectors``).

    Notes
    -----
    - Vector **magnitudes are preserved** by construction. Only directions
      are rotated.
    - For **conformal projections** (stereographic, Mercator, LCC), the
      transform is a pure rotation — results are exact.
    - For **non-conformal projections** (LAEA, equidistant), the Jacobian
      has shear, so the rotation-only approximation introduces a small
      error. This is the same limitation as cartopy's implementation and
      is acceptable for visualization purposes.
    - For **rotated pole** grids, set ``is_rotated_pole=True`` and ensure
      the proj4 string uses ``+lon_0`` (not ``+o_lon_p``) to match
      cartopy's ``RotatedPole`` convention:
      ``+proj=ob_tran +o_proj=longlat +o_lat_p=<P> +lon_0=<L+180>``.

    Examples
    --------
    Transform wind vectors from a polar stereographic grid to lon/lat:

    >>> import numpy as np
    >>> import pyproj
    >>> from transform_vectors import transform_vectors
    >>>
    >>> src  = pyproj.CRS.from_proj4(
    ...     "+proj=stere +lat_0=90 +lon_0=-30 +lat_ts=90 +datum=WGS84")
    >>> tgt  = pyproj.CRS.from_epsg(4326)
    >>>
    >>> x = np.array([[100000, 200000], [100000, 200000]], dtype=float)
    >>> y = np.array([[-800000, -800000], [-700000, -700000]], dtype=float)
    >>> u = np.ones_like(x)   # unit vector pointing in grid-x direction
    >>> v = np.zeros_like(x)
    >>>
    >>> ut, vt = transform_vectors(src, tgt, x, y, u, v)

    Transform vectors on a rotated pole grid:

    >>> src_rp = pyproj.CRS.from_proj4(
    ...     "+proj=ob_tran +o_proj=longlat +o_lat_p=45 +lon_0=270 +datum=WGS84")
    >>> ut, vt = transform_vectors(src_rp, tgt, x_rot, y_rot, u, v,
    ...                            is_rotated_pole=True)
    """
    # ── input validation ──────────────────────────────────────────────────────
    src_crs = CRS.from_user_input(src_crs)
    target_crs = CRS.from_user_input(target_crs)

    x, y, u, v = map(np.asarray, (x, y, u, v))

    if not (x.shape == y.shape == u.shape == v.shape):
        raise ValueError("x, y, u and v must all have the same shape.")
    if x.ndim not in (1, 2):
        raise ValueError("x, y, u and v must be 1D or 2D arrays.")

    # ── detect angular / rotated pole CRS ────────────────────────────────────
    is_angular = is_rotated_pole or (
        src_crs.axis_info[0].unit_name.lower()
        in ("degree", "degree (supplier to define representation)")
    )

    if delta is None:
        delta = 0.001 if is_angular else 1.0

    # ── transformer src → target ──────────────────────────────────────────────
    transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)

    # ── base points in target CRS ─────────────────────────────────────────────
    target_x, target_y = transformer.transform(x, y)

    # ── vector direction and magnitude ────────────────────────────────────────
    vector_magnitudes = np.hypot(u, v)
    vector_angles = np.arctan2(v, u)

    # ── perturbation step in src CRS along vector direction ───────────────────
    x_perturb = np.array(delta * np.cos(vector_angles))
    y_perturb = np.array(delta * np.sin(vector_angles))

    # ── domain boundary handling ──────────────────────────────────────────────
    # Mirror cartopy's approach: if the perturbation takes a point outside
    # the valid domain, reverse the perturbation direction.
    if is_angular:
        x_lo, x_hi = -180.0, 180.0
        y_lo, y_hi = -90.0, 90.0
    else:
        try:
            bounds = src_crs.area_of_use.bounds  # lon_min, lat_min, lon_max, lat_max
            geo_to_src = Transformer.from_crs(
                CRS.from_epsg(4326), src_crs, always_xy=True
            )
            xs, ys = geo_to_src.transform(
                [bounds[0], bounds[2], bounds[0], bounds[2]],
                [bounds[1], bounds[1], bounds[3], bounds[3]],
            )
            x_lo, x_hi = float(np.min(xs)), float(np.max(xs))
            y_lo, y_hi = float(np.min(ys)), float(np.max(ys))
        except Exception:
            x_lo, x_hi = -1e9, 1e9
            y_lo, y_hi = -1e9, 1e9

    eps = 1e-9

    invalid_x = (x + x_perturb < x_lo - eps) | (x + x_perturb > x_hi + eps)
    if invalid_x.any():
        x_perturb[invalid_x] *= -1
        y_perturb[invalid_x] *= -1

    invalid_y = (y + y_perturb < y_lo - eps) | (y + y_perturb > y_hi + eps)
    if invalid_y.any():
        x_perturb[invalid_y] *= -1
        y_perturb[invalid_y] *= -1

    reversed_vectors = np.logical_xor(invalid_x, invalid_y)

    # warn if some corners are still outside the domain
    still_invalid = (x + x_perturb < x_lo - eps) | (x + x_perturb > x_hi + eps)
    if still_invalid.any():
        warnings.warn(
            "Some vectors at source domain corners may not have been "
            "transformed correctly.",
            UserWarning,
            stacklevel=2,
        )

    # ── perturbed points in target CRS ────────────────────────────────────────
    target_x_p, target_y_p = transformer.transform(x + x_perturb, y + y_perturb)

    # ── angle in target CRS ───────────────────────────────────────────────────
    projected_angles = np.arctan2(target_y_p - target_y, target_x_p - target_x)

    if reversed_vectors.any():
        projected_angles[reversed_vectors] += np.pi

    # ── reconstruct vectors preserving magnitude ──────────────────────────────
    ut = vector_magnitudes * np.cos(projected_angles)
    vt = vector_magnitudes * np.sin(projected_angles)

    return ut, vt


def save_transform(
    x,
    y,
    src_proj,
    target_proj,
    delta=None,
    is_rotated_pole=False,
):
    """
    Precompute and store the full 2x2 transformation matrix field for a grid.

    For each grid point, transforms the two unit basis vectors (1,0) and (0,1)
    from ``src_proj`` to ``target_proj`` and assembles the results into a
    transformation matrix T. This matrix can then be reused efficiently to
    transform many different vector fields on the same grid without recomputing
    the projection at every call.

    Parameters
    ----------
    x : 1D np.ndarray
        X coordinates of the grid in ``src_proj`` units (meters or degrees).
    y : 1D np.ndarray
        Y coordinates of the grid in ``src_proj`` units (meters or degrees).
    src_proj : any
        Source CRS. Accepts any format supported by
        ``pyproj.CRS.from_user_input``: EPSG code, proj4 string, WKT, or
        ``pyproj.CRS`` object.
    target_proj : any
        Target CRS. Same formats as ``src_proj``.
    delta : float, optional
        Perturbation step in source CRS native units.
        Defaults to ``1.0`` for metric CRS (meters) and ``0.001`` for
        angular CRS (degrees). Increase for smoother results on coarse
        grids, decrease for higher accuracy on fine grids.
    is_rotated_pole : bool, optional
        Set to ``True`` for rotated pole grids. This forces degree-based
        perturbation and correct domain boundary handling regardless of
        what ``src_crs.axis_info`` reports. Default is ``False``.

    Returns
    -------
    T : np.ndarray, shape (len(y), len(x), 2, 2)
        Transformation matrix field. At each grid point (i, j):

            T[i, j] = | t_i[0]   t_j[0] |   =   | du_target/du_src   du_target/dv_src |
                       | t_i[1]   t_j[1] |       | dv_target/du_src   dv_target/dv_src |

        where ``t_i`` and ``t_j`` are the transformed images of (1,0) and (0,1).

    See Also
    --------
    transform : Apply the precomputed matrix T to a vector field.
    transform_vectors : Transform vectors directly without precomputing T.

    Notes
    -----
    Use this function when you need to transform many different vector fields
    on the same grid — the projection is computed once and reused.
    For a single vector field, calling ``transform_vectors`` directly is
    simpler.

    Examples
    --------
    >>> import numpy as np
    >>> import pyproj
    >>> from transform_vectors import save_transform, transform
    >>>
    >>> src = pyproj.CRS.from_proj4(
    ...     "+proj=stere +lat_0=90 +lon_0=-30 +lat_ts=90 +datum=WGS84")
    >>> tgt = pyproj.CRS.from_epsg(4326)
    >>>
    >>> x = np.linspace(10000, 200000, 50)
    >>> y = np.linspace(-1000000, -500000, 50)
    >>>
    >>> T = save_transform(x, y, src, tgt)   # compute once
    >>>
    >>> wind_u = np.ones((50, 50))
    >>> wind_v = np.zeros((50, 50))
    >>> wind = np.stack([wind_u, wind_v], axis=-1)   # shape (50, 50, 2)
    >>>
    >>> wind_transformed = transform(T, wind)   # apply many times cheaply
    """

    X, Y = np.meshgrid(x, y)

    vect_i = np.stack([np.ones(X.shape), np.zeros(X.shape)], axis=-1)  # (1 0).T
    vect_j = np.stack([np.zeros(X.shape), np.ones(X.shape)], axis=-1)  # (0 1).T
    t_i = transform_vectors(
        src_proj,
        target_proj,
        X,
        Y,
        vect_i[:, :, 0],
        vect_i[:, :, 1],
        delta=delta,
        is_rotated_pole=is_rotated_pole,
    )
    t_j = transform_vectors(
        src_proj,
        target_proj,
        X,
        Y,
        vect_j[:, :, 0],
        vect_j[:, :, 1],
        delta=delta,
        is_rotated_pole=is_rotated_pole,
    )

    T = np.zeros((*X.shape, 2, 2))
    T[..., 0, 0] = t_i[0]
    T[..., 0, 1] = t_j[0]
    T[..., 1, 0] = t_i[1]
    T[..., 1, 1] = t_j[1]

    return T


def transform(T, u):
    """
    Apply a precomputed transformation matrix field to a vector field.

    Performs the matrix-vector multiplication ``T @ u`` at every grid point
    simultaneously using ``np.einsum``. This is the fast path for transforming
    vector fields when the transformation matrix has already been computed by
    ``save_transform``.

    Parameters
    ----------
    T : np.ndarray, shape (..., 2, 2)
        Transformation matrix field as returned by ``save_transform``.
        The leading dimensions must be broadcastable with ``u[..., :2]``.
    u : np.ndarray, shape (..., 2)
        Vector field to transform. The last dimension must be 2 (x and y
        components). Leading dimensions must match those of ``T``.

    Returns
    -------
    np.ndarray, shape (..., 2)
        Transformed vector field. Same shape as ``u``.

    See Also
    --------
    save_transform : Precompute the transformation matrix T for a grid.

    Examples
    --------
    >>> wind = np.stack([wind_u, wind_v], axis=-1)   # shape (ny, nx, 2)
    >>> wind_lonlat = transform(T, wind)              # shape (ny, nx, 2)
    >>>
    >>> # equivalent to looping over grid points:
    >>> # wind_lonlat[i, j] = T[i, j] @ wind[i, j]
    """

    return np.einsum("...ij,...j->...i", T, u)
