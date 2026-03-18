"""
examples/visual_check.py
========================

Visual validation of transform_vectors across all test cases.

Plots three sets of arrows at each grid point:

    Black  — original unit vectors in the source projection
    Red    — transformed vectors (our function) in lon/lat
    Blue   — cartopy transform_vectors (metric CRS only, reference)

Red should overlap black perfectly. Blue should overlap red for
metric projections.

Requirements
------------
    pip install "transform-vectors[plot]"

Usage
-----
    # all cases, 5x5 grid, plot in target CRS (default)
    python examples/visual_check.py

    # specific case numbers (1-based)
    python examples/visual_check.py 1 5 14

    # denser grid (10x10 points)
    python examples/visual_check.py --npts 10

    # choose plot projection: 'source' or 'target'
    python examples/visual_check.py --proj source
    python examples/visual_check.py --proj source --npts 8 1 5
"""

import sys
import os
import argparse

# ensure repo root is on path regardless of where the script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyproj
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from transform_vectors import transform_vectors
from tests.test_cases import TEST_CASES, build_grid


# ── helpers ───────────────────────────────────────────────────────────────────

def build_cartopy_crs(kwargs):
    """Build a cartopy CRS from a kwargs dict, restoring the dict afterwards."""
    cls_name      = kwargs.pop("cls")
    crs           = getattr(ccrs, cls_name)(**kwargs)
    kwargs["cls"] = cls_name
    return crs


def is_geographic(cartopy_crs):
    """Return True if the CRS uses degrees (geographic / rotated pole)."""
    return isinstance(cartopy_crs, (ccrs.PlateCarree, ccrs.Geodetic,
                                    ccrs.RotatedPole))


def central_longitude(lon):
    """
    Circular mean longitude to correctly handle dateline-crossing domains.

    A simple arithmetic mean gives 0° for a grid spanning 170° to -170°,
    while the circular mean correctly gives 180°.

    Parameters
    ----------
    lon : array-like, longitudes in degrees

    Returns
    -------
    float in [-180, 180]
    """
    lon_rad = np.deg2rad(np.asarray(lon))
    return float(np.rad2deg(
        np.arctan2(np.sin(lon_rad).mean(),
                   np.cos(lon_rad).mean())))


def get_axes_crs(lon, plot_proj, cartopy_src):
    """
    Return the cartopy CRS for the map axes.

    For 'target' mode returns PlateCarree centered on the grid's circular-mean
    longitude so that dateline-crossing domains are never split in two.

    Parameters
    ----------
    lon         : 2D array of true longitudes (degrees)
    plot_proj   : 'source' | 'target'
    cartopy_src : cartopy CRS of the source grid
    """
    if plot_proj == 'source':
        return cartopy_src
    return ccrs.PlateCarree(central_longitude=central_longitude(lon))


def compute_extent(x, y, lon, lat, pad, axes_crs, cartopy_src):
    """
    Compute [x_min, x_max, y_min, y_max] and the CRS they are expressed in.

    Padding is interpreted in the correct units automatically:
        geographic axes (PlateCarree, RotatedPole) → degrees
        metric axes (Stereographic, LCC …)         → meters

    Parameters
    ----------
    x, y        : 1D source grid extent arrays  ([x_min, x_max])
    lon, lat    : 2D true geographic coordinates (degrees)
    pad         : padding value from the test case dict
    axes_crs    : cartopy CRS used for the map axes
    cartopy_src : cartopy CRS of the source grid

    Returns
    -------
    extent     : [x_min, x_max, y_min, y_max]
    extent_crs : cartopy CRS the extent values are expressed in
    """
    if is_geographic(axes_crs):
        pad_deg = pad if pad < 50 else 5.0   # safety: if pad is in meters use 5°
        return (
            [lon.min() - pad_deg, lon.max() + pad_deg,
             lat.min() - pad_deg, lat.max() + pad_deg],
            ccrs.PlateCarree()
        )
    else:
        pad_m = pad if pad >= 1 else pad * 111_000  # safety: if pad is in degrees
        return (
            [x[0] - pad_m, x[-1] + pad_m,
             y[0] - pad_m, y[-1] + pad_m],
            cartopy_src
        )


def plot_unit_quiver(ax, x, y, u, v, crs, color,scale=None, label=None, width=0.003,
                    ):
    """Plot vectors normalized to unit length with a fixed display size."""
    ax.quiver(x, y, u, v,
              color=color, transform=crs,
              scale=scale, scale_units='xy',
              width=width, label=label)


# ── single test runner ────────────────────────────────────────────────────────

def run_test(tc, plot_proj='target', npts=5,scale=None):
    """
    Run and plot one test case.

    Parameters
    ----------
    tc        : dict, one entry from TEST_CASES
    plot_proj : 'source' | 'target'
                'source' → map axes in the source (grid) projection,
                           padding in meters
                'target' → map axes in PlateCarree centered on grid,
                           padding in degrees  [default]
    npts      : int
                Number of grid points along each axis. Default 5.
                Increase to see how arrows vary spatially.
    """
    proj4          = tc["proj4"]
    cartopy_kwargs = tc["cartopy_kwargs"].copy()
    pad            = tc["pad"]
    name           = tc["name"]
    is_rp          = tc.get("is_rotated_pole", False)

    cartopy_src = build_cartopy_crs(cartopy_kwargs)
    proj_crs    = pyproj.CRS.from_proj4(proj4)
    geo_crs     = pyproj.CRS.from_epsg(4326)

    # ── build grid at requested resolution ───────────────────────────────────
    X, Y, x, y = build_grid(tc, npts=npts)

    # true lon/lat of all grid points
    t = pyproj.Transformer.from_crs(proj_crs, geo_crs, always_xy=True)
    lon, lat = t.transform(X, Y)

    # ── axes CRS — PlateCarree centered on grid to avoid dateline split ───────
    axes_crs = get_axes_crs(lon, plot_proj, cartopy_src)
    pc_local = ccrs.PlateCarree(central_longitude=central_longitude(lon))
    pc = ccrs.PlateCarree()

    # ── unit vectors in source grid ───────────────────────────────────────────
    u_i = np.ones(X.shape);  v_i = np.zeros(X.shape)
    u_j = np.zeros(X.shape); v_j = np.ones(X.shape)

    # ── our transform ─────────────────────────────────────────────────────────
    ut_i, vt_i = transform_vectors(proj_crs, geo_crs, X, Y, u_i, v_i,
                                    is_rotated_pole=is_rp)
    ut_j, vt_j = transform_vectors(proj_crs, geo_crs, X, Y, u_j, v_j,
                                    is_rotated_pole=is_rp)

    # ── cartopy reference (metric CRS only) ───────────────────────────────────
    cu_i, cv_i = pc_local.transform_vectors(cartopy_src, X, Y, u_i, v_i)
    cu_j, cv_j = pc_local.transform_vectors(cartopy_src, X, Y, u_j, v_j)


    # ── extent ────────────────────────────────────────────────────────────────
    extent, extent_crs = compute_extent(x, y, lon, lat, pad, axes_crs,
                                         cartopy_src)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 7))
    ax  = plt.axes(projection=axes_crs)

    try:
        ax.set_extent(extent, crs=extent_crs)
    except Exception:
        ax.set_global()

    ax.add_feature(cfeature.LAND,  zorder=0, edgecolor='black',
                   facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='lightblue')
    ax.coastlines(resolution='50m', color='black', linewidth=0.8)

    # grid border in source CRS
    for seg in [(X[0,:],Y[0,:]), (X[-1,:],Y[-1,:]),
                (X[:,0],Y[:,0]), (X[:,-1],Y[:,-1])]:
        ax.plot(seg[0], seg[1], color='blue', lw=1, transform=cartopy_src)

    # black — source grid unit vectors
    plot_unit_quiver(ax, X, Y, u_i, v_i, cartopy_src,
                     'black',scale=scale,label='e_x/e_y (proj)', width=0.009)
    plot_unit_quiver(ax, X, Y, u_j, v_j, cartopy_src,
                     'black',scale=scale, width=0.009)
    # blue — cartopy reference 
    plot_unit_quiver(ax, lon, lat, cu_i, cv_i, pc,
                        'blue',scale=scale, label='e_x/e_y (cartopy ref)', width=0.006)
    plot_unit_quiver(ax, lon, lat, cu_j, cv_j, pc,
                        'blue',scale=scale, width=0.006)
    # red — our transform
    plot_unit_quiver(ax, lon, lat, ut_i, vt_i, pc,
                     'red',scale=scale, label='e_x/e_y (transform_vectors)')
    plot_unit_quiver(ax, lon, lat, ut_j, vt_j, pc, 'red',scale=scale)

    

    ax.gridlines(draw_labels=True, linewidth=0.5, linestyle='--', color='gray')
    ax.legend(loc='lower left', fontsize=8)

    note     = tc.get("note", "")
    proj_lbl = f"[plot: {plot_proj}, {npts}x{npts} grid]"
    plt.title(f"{name}  {proj_lbl}\n{note}", fontsize=10)
    plt.tight_layout()
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual validation of transform_vectors across projection test cases.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "cases", nargs="*", type=int,
        help="Case numbers to run (1-based). Omit to run all cases."
    )
    parser.add_argument(
        "--proj", choices=["source", "target"], default="target",
        help=(
            "Map projection used for the axes:\n"
            "  target (default) — PlateCarree centered on the grid,\n"
            "                     padding in degrees\n"
            "  source           — source grid projection,\n"
            "                     padding in meters"
        )
    )
    parser.add_argument(
        "--npts", type=int, default=5, metavar="N",
        help=(
            "Number of grid points along each axis (default: 5).\n"
            "Increase to see how the transform varies spatially.\n"
            "Examples: --npts 10  --npts 20"
        )
    )
    parser.add_argument(
        "--scale", type=float, default=30, metavar="S",
        help=(
            "Arrow display scale (default: 30).\n"
            "Passed to matplotlib quiver with scale_units='inches'.\n"
            "Higher value → shorter arrows.\n"
            "Lower value  → longer arrows.\n"
            "Examples: --scale 10  --scale 60"
        )
    )
    args = parser.parse_args()
 
    if args.npts < 2:
        parser.error("--npts must be >= 2")
    if args.scale <= 0:
        parser.error("--scale must be > 0")
 
    cases = ([TEST_CASES[i - 1] for i in args.cases]
             if args.cases else TEST_CASES)
 
    for tc in cases:
        run_test(tc, plot_proj=args.proj, npts=args.npts, scale=args.scale)